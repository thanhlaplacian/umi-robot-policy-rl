#!/usr/bin/env python3
"""Diagnose where fork-vs-RLinf action differences come from.

The fork's serving path (`Gr00tPolicy`) tokenizes with `processor(messages)` + `collate_fn`;
RLinf calls the processor's `process_observation`. This script builds both input sets for the
same observation, compares them tensor by tensor, then runs the SAME model.get_action (same
seed) on each and reports the action difference each input set produces.
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cfg", default=os.path.join(ROOT, "configs/model/gr00t_n1d7_umi.yaml"))
    ap.add_argument("--dataset", default="/data/dataset/26-W33-TELE2-rebot-batch1")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--step", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)

    import umi_rl.model  # noqa: F401
    from umi_rl import converters
    from rlinf.models import get_model
    import gr00t.model  # noqa: F401
    from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
    from umi_data_sdk.core.embodiment_tags import EmbodimentTag
    from umi_data_sdk.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from umi_data_sdk.dataset.sharded_single_step_dataset import extract_step_data

    cfg = OmegaConf.load(a.model_cfg)
    tag = EmbodimentTag(cfg.embodiment_tag)
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=cfg.model_path, device="cuda")
    rl = get_model(cfg); rl.eval()
    modality = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(a.dataset, modality_configs=modality, video_backend="pyav")
    dp = extract_step_data(loader.get_episode(a.episode), a.step, modality, tag)
    views = [k for k in modality["video"].modality_keys if k in dp.images]

    # fork inputs (serving path)
    fork_obs = {"video": {v: np.asarray(dp.images[v])[None].astype(np.uint8) for v in views},
                "state": {k: np.asarray(dp.states[k], dtype=np.float32)[None] for k, _ in converters.STATE_KEYS},
                "language": {converters.LANGUAGE_KEY: [[dp.text]]}}
    from umi_data_sdk.core.types import MessageType
    unb = policy._unbatch_observation(fork_obs)
    processed = []
    for o in unb:  # exactly Gr00tPolicy._get_action steps 1-2 (the deployment path)
        vla_step_data = policy._to_vla_step_data(o)
        processed.append(policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]))
    fork_in = _rec_to_dtype(policy.collate_fn(processed), dtype=torch.bfloat16)

    # rlinf inputs
    env_obs = {"states": torch.from_numpy(np.concatenate([fork_obs["state"][k][:, 0] for k, _ in converters.STATE_KEYS], -1)),
               "task_descriptions": [dp.text]}
    env_key = {v: e for e, v in converters.VIEW_KEYS}
    for v in views:
        env_obs[env_key[v]] = torch.from_numpy(fork_obs["video"][v][:, 0])
    _, obs_copy, _ = rl._prepare_rollout_observation(env_obs)
    rl_in = rl.apply_transforms(obs_copy)
    rl_in = rl._cast_float_tensors_to_compute_dtype(rl_in, rl.compute_dtype)

    def flat(d):
        out = {}
        for k, v in (d.items() if isinstance(d, dict) else d.data.items()):
            if isinstance(v, dict) or hasattr(v, "data"):
                out.update({f"{k}.{kk}": vv for kk, vv in flat(v).items()})
            else:
                out[k] = v
        return out
    fi, ri = flat(fork_in), flat(rl_in)
    print("fork keys :", sorted(fi)); print("rlinf keys:", sorted(ri))
    for k in sorted(set(fi) & set(ri)):
        x, y = fi[k], ri[k]
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            print(f"{k:22s} non-tensor fork={type(x).__name__} rlinf={type(y).__name__}"); continue
        x, y = x.detach().cpu(), y.detach().cpu()
        if x.shape != y.shape:
            print(f"{k:22s} SHAPE fork={tuple(x.shape)} rlinf={tuple(y.shape)} dtype {x.dtype}/{y.dtype}"); continue
        if x.dtype.is_floating_point:
            d = (x.float() - y.float()).abs(); print(f"{k:22s} shape={tuple(x.shape)} max|Δ|={d.max():.3e} mean|Δ|={d.mean():.3e} |x|={x.float().abs().mean():.3e} dtype {x.dtype}/{y.dtype}")
        else:
            print(f"{k:22s} shape={tuple(x.shape)} equal={bool((x==y).all())} dtype {x.dtype}/{y.dtype}")

    # same model, same seed, each input set
    def run(inp):
        inp = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in (inp.items() if isinstance(inp, dict) else inp.data.items())}
        torch.manual_seed(a.seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return policy.model.get_action(inp)["action_pred"][:, :16, :20].float().cpu()
    af, ar = run(fork_in), run(rl_in)
    print(f"\nnormalized action_pred[:, :16, :20]: fork-inputs vs rlinf-inputs through the SAME fork model: max|Δ|={(af-ar).abs().max():.3e} mean|Δ|={(af-ar).abs().mean():.3e} |a|={af.abs().mean():.3e}")
    # and the RLinf model object with the fork's inputs (weights identical?)
    torch.manual_seed(a.seed)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inp = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in fork_in.items()}
        arl = rl.get_action(inp)["action_pred"][:, :16, :20].float().cpu()
    print(f"fork inputs through fork model vs RLinf model object: max|Δ|={(af-arl).abs().max():.3e}")


if __name__ == "__main__":
    main()
