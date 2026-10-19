#!/usr/bin/env python3
"""W0 parity: the umi_wam eval path (RLinf predict_action_batch, mode eval) against the fork's own Gr00tPolicy on the
same model and the same simulator observations. Both draw the initial noise from torch's global RNG, so seeding right
before each call makes them comparable; the criterion is max |diff| = 0.0 on the decoded 14-dim action. A second fork
call with seed+1 gives the noise-to-noise spread as a scale reference.

usage: wam_parity_check.py [--model-cfg CFG] [--ckpt DIR] [--n-envs 4] [--seed 0] [--refresh-obs]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from wam_common import DEFAULT_MODEL_CFG, fork_obs_from, load_model, sim_batches


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh-obs", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from umi_rl import converters

    t0 = time.time()
    model, cfg = load_model(a.model_cfg, a.ckpt)
    model.eval()
    print(f"[umi_wam] {type(model).__name__} loaded in {time.time()-t0:.0f}s; value head: {model.value_head is not None}", flush=True)
    policy = Gr00tPolicy.from_components(model, model.processor, model.embodiment_tag, strict=False)
    batches = sim_batches(model, n_envs=a.n_envs, refresh=a.refresh_obs)
    report = []
    for bi, env_obs in enumerate(batches):
        fobs = fork_obs_from(env_obs, model._video_keys, model._state_keys)
        torch.manual_seed(a.seed)
        fork, _ = policy.get_action(fobs)
        torch.manual_seed(a.seed + 1)
        fork2, _ = policy.get_action(fobs)
        torch.manual_seed(a.seed)
        with torch.no_grad():
            rl, res = model.predict_action_batch(env_obs=env_obs, mode="eval")
        f1 = converters.convert_to_umi_action_n1d7(fork, chunk_size=cfg.num_action_chunks)
        f2 = converters.convert_to_umi_action_n1d7(fork2, chunk_size=cfg.num_action_chunks)
        rl = np.asarray(rl, dtype=np.float32)
        d, spread = float(np.abs(f1 - rl).max()), float(np.abs(f1 - f2).max())
        print(f"batch {bi} ({rl.shape}): max|fork - umi_wam eval| = {d:.3e}   noise spread (seed vs seed+1) = {spread:.3e}", flush=True)
        report.append({"batch": bi, "shape": list(rl.shape), "max_abs_diff": d, "seed_spread": spread,
                       "result_keys": sorted(k for k, v in res.items() if v is not None)})
    ok = all(r["max_abs_diff"] == 0.0 for r in report)
    print("PARITY", "OK (bit-identical)" if ok else "FAILED", flush=True)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
