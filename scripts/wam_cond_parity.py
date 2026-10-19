#!/usr/bin/env python3
"""W1 check: the umi_wam train-mode sampler (the PPO rollout path) at noise_level 0 must reproduce the fork's
get_action (the eval path) on the same initial noise: same observation K/V, same head conditioning (progress /
arm-active), same Euler steps. The train path keeps the chain in fp32, so the expected gap is bf16 rounding, not 0.
The PHASE3 lesson: a dropped conditioning row showed up here as a 2e-2 gap.

usage: wam_cond_parity.py [--model-cfg CFG] [--seed 0]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from wam_common import DEFAULT_MODEL_CFG, load_model, sim_batches


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    model, cfg = load_model(a.model_cfg)
    model.eval()
    model.rl_config["noise_level"] = 0.0
    worst = 0.0
    for bi, env_obs in enumerate(sim_batches(model)):
        inputs, states = model._collate(env_obs)
        torch.manual_seed(a.seed)
        with torch.inference_mode():
            ev = model.get_action(inputs)["action_pred"].float()[..., : model.n_real]
        torch.manual_seed(a.seed)
        tr, res = model._rl_sample(inputs, "train")
        torch.manual_seed(a.seed + 1)
        with torch.inference_mode():
            ev2 = model.get_action(inputs)["action_pred"].float()[..., : model.n_real]
        d_norm = float((ev - tr).abs().max())
        spread = float((ev - ev2).abs().max())
        dec_e, dec_t = model._decode(ev, states), model._decode(tr, states)
        d_dec = float(np.abs(dec_e - dec_t).max())
        lp = res["prev_logprobs"]
        print(f"batch {bi}: normalized max|eval - train(noise 0)| = {d_norm:.3e} (|a| max {float(ev.abs().max()):.2f}); "
              f"decoded {d_dec:.3e}; seed spread {spread:.3e}; prev_logprobs all zero: {bool((lp == 0).all())}", flush=True)
        worst = max(worst, d_norm)
    print("COND PARITY", "OK" if worst < 2e-2 else "FAILED", f"(worst normalized gap {worst:.3e})", flush=True)


if __name__ == "__main__":
    main()
