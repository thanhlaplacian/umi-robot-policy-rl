#!/usr/bin/env python3
"""Phase-0 parity check: the same dataset observation through the fork's own Gr00tPolicy and
through RLinf's RL wrapper (model_type gr00t_n1d7_umi) must give the same decoded action.

Both paths draw the flow-matching noise from torch's global RNG, so seeding right before each
call makes the comparison deterministic. Reports per-key max |diff| between the two paths and,
as a sanity anchor, each path's distance to the ground-truth chunk.

usage: parity_check.py [--ckpt DIR] [--dataset DIR] [--episode N] [--steps 0,40,80] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-cfg", default=os.path.join(ROOT, "configs/model/gr00t_n1d7_umi.yaml"))
    ap.add_argument("--ckpt", default=None, help="override model_path from the model config")
    ap.add_argument("--dataset", default="/data/dataset/26-W36-TELE2-rebot-batch1")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--steps", default="0,40,80")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--out", default=None, help="write the numbers as json")
    a = ap.parse_args()
    steps = [int(s) for s in a.steps.split(",")]

    import umi_rl.model  # registers gr00t_n1d7_umi + converters
    from umi_rl import converters
    from rlinf.models import get_model
    import gr00t.model  # noqa: F401  registers Gr00tN1d7 with Auto*
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from umi_data_sdk.core.embodiment_tags import EmbodimentTag
    from umi_data_sdk.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from umi_data_sdk.dataset.sharded_single_step_dataset import extract_step_data

    cfg = OmegaConf.load(a.model_cfg)
    if a.ckpt:
        cfg.model_path = a.ckpt
    tag = EmbodimentTag(cfg.embodiment_tag)

    t0 = time.time()
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=cfg.model_path, device="cuda")
    print(f"[fork] Gr00tPolicy loaded in {time.time()-t0:.0f}s")
    t0 = time.time()
    rl_model = get_model(cfg)
    rl_model.eval()
    print(f"[rlinf] {type(rl_model).__name__} loaded in {time.time()-t0:.0f}s")

    modality = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(a.dataset, modality_configs=modality, video_backend=a.video_backend)
    traj = loader.get_episode(a.episode)
    print(f"dataset {a.dataset} episode {a.episode}: {len(traj)} frames")

    # ---- build the two observation formats for a batch of `steps`
    dps = [extract_step_data(traj, s, modality, tag) for s in steps]
    views = [k for k in modality["video"].modality_keys if k in dps[0].images]
    fork_obs = {"video": {}, "state": {}, "language": {}}
    for v in views:
        fork_obs["video"][v] = np.stack([np.asarray(dp.images[v]) for dp in dps]).astype(np.uint8)  # (B,T,H,W,3)
    for k, _ in converters.STATE_KEYS:
        fork_obs["state"][k] = np.stack([np.asarray(dp.states[k], dtype=np.float32) for dp in dps])  # (B,T,D)
    fork_obs["language"][converters.LANGUAGE_KEY] = [[dp.text] for dp in dps]
    gt = {k: np.stack([np.asarray(dp.actions[k], dtype=np.float32) for dp in dps]) for k, _ in converters.ACTION_KEYS}

    B = len(steps)
    env_obs = {"states": torch.from_numpy(np.concatenate([fork_obs["state"][k][:, 0] for k, _ in converters.STATE_KEYS], -1)),
               "task_descriptions": [dp.text for dp in dps]}
    env_key = {v: e for e, v in converters.VIEW_KEYS}
    for v in views:
        env_obs[env_key[v]] = torch.from_numpy(fork_obs["video"][v][:, 0])  # (B,H,W,3)
    print(f"views: {views}  batch: {B}  prompt: {dps[0].text!r}")

    # ---- fork path
    torch.manual_seed(a.seed)
    fork_action, _ = policy.get_action(fork_obs)
    # ---- rlinf path
    torch.manual_seed(a.seed)
    with torch.no_grad():
        rl_raw, _ = rl_model.predict_action_batch(env_obs=env_obs, mode="eval")
    rl_raw = np.asarray(rl_raw, dtype=np.float32)  # (B, 16, 14)
    fork_flat = converters.convert_to_umi_action_n1d7(fork_action, chunk_size=cfg.num_action_chunks)
    gt_flat = converters.convert_to_umi_action_n1d7(gt, chunk_size=cfg.num_action_chunks)

    rows = []
    off = 0
    for key, dim in converters.ACTION_KEYS:
        sl = slice(off, off + dim); off += dim
        d = np.abs(fork_flat[..., sl] - rl_raw[..., sl])
        rows.append({"key": key, "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                     "scale_|a|": float(np.abs(fork_flat[..., sl]).mean()),
                     "fork_vs_gt": float(np.abs(fork_flat[..., sl] - gt_flat[..., sl]).mean()),
                     "rlinf_vs_gt": float(np.abs(rl_raw[..., sl] - gt_flat[..., sl]).mean())})
    print(f"\n{'key':30s} {'max|Δ|':>10s} {'mean|Δ|':>10s} {'mean|a|':>10s} {'fork-GT':>10s} {'rlinf-GT':>10s}")
    for r in rows:
        print(f"{r['key']:30s} {r['max_abs_diff']:10.2e} {r['mean_abs_diff']:10.2e} {r['scale_|a|']:10.2e} {r['fork_vs_gt']:10.2e} {r['rlinf_vs_gt']:10.2e}")
    worst = max(r["max_abs_diff"] for r in rows)
    print(f"\nshapes: fork {fork_flat.shape} rlinf {rl_raw.shape}   worst max|Δ| = {worst:.3e}")
    if a.out:
        json.dump({"ckpt": cfg.model_path, "dataset": a.dataset, "episode": a.episode, "steps": steps, "seed": a.seed,
                   "views": views, "rows": rows, "worst_max_abs_diff": worst}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
