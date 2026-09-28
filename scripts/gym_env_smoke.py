#!/usr/bin/env python3
"""Smoke of LaplacianRLEnv: reset N envs, hold still, then drive the company policy closed-loop
for a few chunks. Prints timings per phase and saves camera frames. Run inside the container:

  docker exec -e CUDA_VISIBLE_DEVICES=7 umi-thanh-maskfix bash -lc 'cd <repo> && <venv>/bin/python scripts/gym_env_smoke.py --num-envs 2 --chunks 3'
"""
from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_frames(obs, out_dir, tag):
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    for k, v in obs.items():
        if k == "states":
            continue
        for i in range(v.shape[0]):
            Image.fromarray(v[i].cpu().numpy()).save(os.path.join(out_dir, f"{tag}_env{i}_{k}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=3, help="policy chunks (16 steps each) to run; 0 = no policy")
    ap.add_argument("--hold-steps", type=int, default=5)
    ap.add_argument("--out", default="/home/thanh/rlinf-runs/logs/umi-rl/gym_smoke")
    ap.add_argument("--model-cfg", default=os.path.join(ROOT, "configs/model/gr00t_n1d7_umi.yaml"))
    ap.add_argument("--prompt", default="Pick up the objects on the shelf and place them in the basket")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
    from umi_rl.envs.laplacian.rl_env import LaplacianRLConfig, LaplacianRLEnv

    t = time.time()
    env = LaplacianRLEnv(LaplacianRLConfig(seed=0), num_envs=a.num_envs, device="cuda:0", render=not a.no_render)
    print(f"env built in {time.time()-t:.1f}s  actuators={len(env.env.action_names)} objects/env={len(env.slots)}")
    t = time.time(); obs = env.reset(); torch.cuda.synchronize()
    print(f"reset in {time.time()-t:.1f}s  state[0]={np.round(obs['states'][0].cpu().numpy(), 3).tolist()}")
    for k, v in obs.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    if not a.no_render:
        save_frames(obs, a.out, "reset")
    # hold: zero deltas, grippers open -> the arms must stay put (IK identity)
    s0 = obs["states"].clone()
    zero = torch.zeros(a.num_envs, 14, device="cuda:0")
    for k in range(a.hold_steps):
        t = time.time(); obs, r, term, trunc, info = env.step(zero); torch.cuda.synchronize()
        print(f"hold step {k}: {time.time()-t:.3f}s  {env.timing}  drift={float((obs['states'][:,1:4]-s0[:,1:4]).abs().max()):.4f} m")
    # a scripted move: 2 cm forward along the EE x axis per step for 5 steps, then back
    for sign in (1, -1):
        for k in range(5):
            act = zero.clone(); act[:, 1] = 0.02 * sign; act[:, 8] = 0.02 * sign
            obs, r, term, trunc, info = env.step(act)
        print(f"scripted move sign={sign}: state[0] pos_R={np.round(obs['states'][0,1:4].cpu().numpy(),3).tolist()}")
    if not a.no_render:
        save_frames(obs, a.out, "moved")
    if a.chunks > 0:
        from omegaconf import OmegaConf
        import umi_rl.model  # noqa
        from rlinf.models import get_model
        cfg = OmegaConf.load(a.model_cfg)
        t = time.time(); model = get_model(cfg); model.eval(); print(f"policy loaded in {time.time()-t:.0f}s")
        for c in range(a.chunks):
            env_obs = {"states": obs["states"].cpu(), "task_descriptions": [a.prompt] * a.num_envs,
                       "wrist_left_images": obs["cam_wrist_left"], "wrist_right_images": obs["cam_wrist_right"]}
            if "cam_head" in obs:
                env_obs["head_images"] = obs["cam_head"]
            t = time.time()
            with torch.no_grad():
                actions, _ = model.predict_action_batch(env_obs=env_obs, mode="eval")  # [N,16,14]
            t_pol = time.time() - t
            actions = torch.as_tensor(np.asarray(actions), device="cuda:0")
            t = time.time()
            for s in range(actions.shape[1]):
                obs, r, term, trunc, info = env.step(actions[:, s])
            torch.cuda.synchronize()
            print(f"chunk {c}: policy {t_pol:.2f}s, 16 env steps {time.time()-t:.2f}s, |a| pos={float(actions[...,1:4].abs().mean()):.4f} m  grip={actions[:,-1,0].tolist()}  success={info['success'].tolist()}  min_dist={np.round(env.min_tcp_object_distance().cpu().numpy(),3).tolist()}")
        if not a.no_render:
            save_frames(obs, a.out, f"chunk{a.chunks}")
    print("done; frames in", a.out)


if __name__ == "__main__":
    main()
