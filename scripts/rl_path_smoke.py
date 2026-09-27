#!/usr/bin/env python3
"""Phase-0 smoke of the RL surface on the company model: rollout in ``train`` mode (flow-SDE
chains, log-probs, values) and the actor forward that recomputes log-probs/values/entropy from
the stored chains, with action_dim 20 x 16 chunks. Single process, one GPU, no Ray."""
from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cfg", default=os.path.join(ROOT, "configs/model/gr00t_n1d7_umi.yaml"))
    ap.add_argument("--dataset", default="/data/dataset/26-W33-TELE2-rebot-batch1")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--steps", default="0,40,80,120")
    a = ap.parse_args()
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
    import umi_rl.model  # noqa: F401
    from umi_rl import converters
    from rlinf.models import get_model
    import gr00t.model  # noqa: F401
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from umi_data_sdk.core.embodiment_tags import EmbodimentTag
    from umi_data_sdk.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from umi_data_sdk.dataset.sharded_single_step_dataset import extract_step_data

    cfg = OmegaConf.load(a.model_cfg)
    tag = EmbodimentTag(cfg.embodiment_tag)
    model = get_model(cfg)
    proc = Gr00tN1d7Processor.from_pretrained(cfg.model_path)
    modality = proc.modality_configs[tag.value]
    loader = LeRobotEpisodeLoader(a.dataset, modality_configs=modality, video_backend="pyav")
    traj = loader.get_episode(a.episode)
    steps = [int(s) for s in a.steps.split(",")]
    dps = [extract_step_data(traj, s, modality, tag) for s in steps]
    views = [k for k in modality["video"].modality_keys if k in dps[0].images]
    env_obs = {"states": torch.from_numpy(np.stack([np.concatenate([np.asarray(dp.states[k], np.float32)[0] for k, _ in converters.STATE_KEYS]) for dp in dps])),
               "task_descriptions": [dp.text for dp in dps]}
    env_key = {v: e for e, v in converters.VIEW_KEYS}
    for v in views:
        env_obs[env_key[v]] = torch.from_numpy(np.stack([np.asarray(dp.images[v])[0] for dp in dps]).astype(np.uint8))
    B = len(steps)

    # ---- rollout, train mode: chains + prev_logprobs + prev_values
    model.eval()
    torch.manual_seed(0)
    t = time.time()
    with torch.no_grad():
        actions, result = model.predict_action_batch(env_obs=env_obs, mode="train")
    print(f"rollout(train) {time.time()-t:.1f}s  actions {np.asarray(actions).shape}  keys {sorted(result)}")
    fi = result["forward_inputs"]
    for k, v in fi.items():
        if torch.is_tensor(v):
            print(f"  forward_inputs[{k}] {tuple(v.shape)} {v.dtype}")
    pl, pv = result["prev_logprobs"], result["prev_values"]
    print(f"  prev_logprobs {tuple(pl.shape)} finite={bool(torch.isfinite(pl).all())} mean={pl.float().mean():.3f}")
    print(f"  prev_values   {tuple(pv.shape)} finite={bool(torch.isfinite(pv).all())}")

    # ---- actor forward on the stored chains (what the PPO update calls)
    model.train()
    t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(forward_inputs=fi, prev_logprobs=pl, compute_logprobs=True, compute_values=True)
    print(f"actor forward {time.time()-t:.1f}s  keys {sorted(out)}")
    lp = out["logprobs"]
    print(f"  logprobs {tuple(lp.shape)} finite={bool(torch.isfinite(lp).all())}  |logprobs - prev_logprobs| max={(lp.float()-pl.float().to(lp.device)).abs().max():.3e}")
    print(f"  values {tuple(out['values'].shape)}  entropy {tuple(out['entropy'].shape) if torch.is_tensor(out['entropy']) else out['entropy']}")
    # backward through the ratio to prove gradients flow into backbone/head (no optimizer step)
    ratio = torch.exp(lp.float() - pl.float().to(lp.device))
    loss = -(ratio.mean()) + out["values"].float().pow(2).mean()
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  backward ok: {n_grad} parameter tensors received non-zero grad; ratio mean {ratio.mean():.4f}")
    print(f"  peak GPU mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB for B={B}")


if __name__ == "__main__":
    main()
