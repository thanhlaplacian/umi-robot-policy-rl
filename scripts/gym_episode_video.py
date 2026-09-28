#!/usr/bin/env python3
"""Run one full episode of the company policy in LaplacianRLEnv, write an MP4 (wrist L | wrist R |
head, tiled) with a reward/step overlay, and dump per-step rewards, cumulative reward and the
TCP-to-object distance to JSON and a PNG curve."""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tile(obs, i=0):
    from PIL import Image

    ims = [Image.fromarray(obs[k][i].cpu().numpy()) for k in ("cam_wrist_left", "cam_wrist_right", "cam_head") if k in obs]
    H = max(im.height for im in ims)
    out = Image.new("RGB", (sum(im.width for im in ims), H))
    x = 0
    for im in ims:
        out.paste(im, (x, (H - im.height) // 2)); x += im.width
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/thanh/rlinf-runs/logs/umi-rl/gym_episode")
    ap.add_argument("--max-steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dense-reward", type=float, default=0.0, help="env reward weight on -min TCP-object distance (0 = sparse success only)")
    ap.add_argument("--model-cfg", default=os.path.join(ROOT, "configs/model/gr00t_n1d7_umi.yaml"))
    ap.add_argument("--ckpt", default=None, help="override model_path of the model config")
    ap.add_argument("--tag", default="seed", help="output name tag")
    ap.add_argument("--prompt", default="Pick up the objects on the shelf and place them in the basket")
    a = ap.parse_args()
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
    os.makedirs(a.out, exist_ok=True)
    from PIL import ImageDraw
    from omegaconf import OmegaConf
    import umi_rl.model  # noqa
    from rlinf.models import get_model
    from umi_rl.envs.laplacian.rl_env import LaplacianRLConfig, LaplacianRLEnv

    env = LaplacianRLEnv(LaplacianRLConfig(seed=a.seed, max_episode_steps=a.max_steps, dense_reward=a.dense_reward), num_envs=1, device="cuda:0")
    mcfg = OmegaConf.load(a.model_cfg)
    if a.ckpt:
        mcfg.model_path = a.ckpt
    model = get_model(mcfg); model.eval()
    obs = env.reset(seeds=[a.seed])
    frames, log = [], []
    cum = 0.0
    step, done = 0, False
    t_start = time.time()
    while not done and step < a.max_steps:
        env_obs = {"states": obs["states"].cpu(), "task_descriptions": [a.prompt],
                   "wrist_left_images": obs["cam_wrist_left"], "wrist_right_images": obs["cam_wrist_right"], "head_images": obs.get("cam_head")}
        with torch.no_grad():
            actions, _ = model.predict_action_batch(env_obs=env_obs, mode="eval")
        actions = torch.as_tensor(np.asarray(actions), device="cuda:0")[0]
        for s in range(actions.shape[0]):
            obs, r, term, trunc, info = env.step(actions[s][None])
            r = float(r[0]); cum += r; step += 1
            dist = float(env.min_tcp_object_distance()[0])
            objs = env.object_positions()[0].cpu().numpy().round(3).tolist()
            log.append({"step": step, "t_s": step / 15, "reward": r, "cumulative_reward": cum, "min_tcp_object_dist_m": dist,
                        "grip": [float(actions[s, 0]), float(actions[s, 7])], "success": bool(info["success"][0]), "objects_xyz": objs})
            im = tile(obs); d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 560, 22], fill=(0, 0, 0))
            d.text((4, 4), f"t={step/15:5.2f}s step={step:3d}  r={r:+.3f}  R={cum:+.3f}  min_dist={dist:.3f} m  success={bool(info['success'][0])}", fill=(255, 255, 0))
            frames.append(np.asarray(im))
            done = bool(term[0] or trunc[0])
            if done:
                break
    wall = time.time() - t_start
    # video (PyAV; h264 if available, else mpeg4)
    import av
    h, w = frames[0].shape[:2]
    path = os.path.join(a.out, f"episode_{a.tag}{a.seed}.mp4")
    with av.open(path, "w") as cont:
        try:
            stream = cont.add_stream("libx264", rate=15); stream.pix_fmt = "yuv420p"; stream.options = {"crf": "20"}
        except Exception:
            stream = cont.add_stream("mpeg4", rate=15); stream.pix_fmt = "yuv420p"; stream.bit_rate = 8_000_000
        stream.width, stream.height = w - w % 2, h - h % 2
        for f in frames:
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f[: stream.height, : stream.width]), format="rgb24")
            for pkt in stream.encode(vf):
                cont.mux(pkt)
        for pkt in stream.encode():
            cont.mux(pkt)
    json.dump({"seed": a.seed, "model_path": str(mcfg.model_path), "steps": step, "duration_s": step / 15, "wall_s": wall, "cumulative_reward": cum, "success": log[-1]["success"],
               "dense_reward_weight": a.dense_reward, "log": log}, open(os.path.join(a.out, f"episode_{a.tag}{a.seed}.json"), "w"), indent=1)
    # curve
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = [x["t_s"] for x in log]
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax[0].plot(t, [x["cumulative_reward"] for x in log], color="#2a78d6", lw=2); ax[0].set_ylabel("cumulative reward"); ax[0].grid(alpha=.3)
    ax[1].plot(t, [x["min_tcp_object_dist_m"] for x in log], color="#eb6834", lw=2); ax[1].set_ylabel("min TCP-object dist (m)"); ax[1].set_xlabel("time (s) @15 Hz"); ax[1].grid(alpha=.3)
    ax[0].set_title(f"{os.path.basename(os.path.dirname(str(mcfg.model_path)))} seed {a.seed}: {step} steps, success={log[-1]['success']}, cumulative reward {cum:+.3f}")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"episode_{a.tag}{a.seed}_reward.png"), dpi=110)
    print(f"steps={step} ({step/15:.1f}s sim, {wall:.0f}s wall)  success={log[-1]['success']}  cumulative_reward={cum:+.3f}  final_min_dist={log[-1]['min_tcp_object_dist_m']:.3f}  video={path}")


if __name__ == "__main__":
    main()
