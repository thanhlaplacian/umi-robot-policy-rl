#!/usr/bin/env python3
"""W1 check: rollout then replay of the same chains with the same weights (one GPU, no Ray).

* rollout: predict_action_batch(mode train) -> chains [B,5,16,20], prev_logprobs [B,16,20] (the noisy step), prev_values [B,1];
* replay: default_forward in train mode with grad -> exp(logprobs - prev_logprobs) must be 1 (within 1e-3);
* a PPO-shaped loss backward: finite grads, timing and peak memory at the given micro-batch, with and without
  gradient checkpointing.

usage: wam_rl_path_smoke.py [--micro-batch 8] [--rounds 3]
"""
from __future__ import annotations

import argparse
import time

import torch

from wam_common import DEFAULT_MODEL_CFG, load_model, sim_batches


def cat_obs(batches, n):
    out = {}
    for k in batches[0]:
        v = [b[k] for b in batches]
        out[k] = sum(v, []) if isinstance(v[0], list) else torch.cat(v, 0)
    reps = -(-n // out["states"].shape[0])
    for k in out:
        out[k] = (out[k] * reps)[:n] if isinstance(out[k], list) else out[k].repeat(reps, *[1] * (out[k].dim() - 1))[:n]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    model, cfg = load_model(a.model_cfg)
    env_obs = cat_obs(sim_batches(model), a.micro_batch)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params {sum(p.numel() for p in params)/1e9:.3f} B (value head {sum(p.numel() for p in model.value_head.parameters())/1e6:.2f} M)", flush=True)
    ok = True
    for ckpt in (False, True):
        if ckpt:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        for r in range(a.rounds):
            model.eval()
            t0 = time.time()
            actions, res = model.predict_action_batch(env_obs=env_obs, mode="train")
            torch.cuda.synchronize(); t_roll = time.time() - t0
            fi = res["forward_inputs"]
            idx = int(fi["denoise_inds"][0, 0])
            model.train(); model.set_frozen_modules_to_eval_mode()
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
            out = model(forward_inputs=fi, compute_values=True)
            ratio = torch.exp(out["logprobs"] - res["prev_logprobs"].to(out["logprobs"].device))
            adv = torch.randn(ratio.shape[0], device=ratio.device)
            loss = -(ratio.mean(dim=(1, 2)) * adv).mean() + ((out["values"] - 1.0) ** 2).mean()
            loss.backward()
            torch.cuda.synchronize(); t_upd = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            grads_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
            n_grad = sum(p.grad is not None for p in params)
            dev = float((ratio - 1).abs().max())
            vgap = float((out["values"].detach() - res["prev_values"][:, 0].to(out["values"].device)).abs().max())
            print(f"ckpt={ckpt} round {r}: denoise idx {idx}; actions {tuple(actions.shape)}; chains {tuple(fi['chains'].shape)}; "
                  f"prev_logprobs {tuple(res['prev_logprobs'].shape)} (sum {float(res['prev_logprobs'].sum()):.1f}); "
                  f"max|ratio-1| {dev:.2e}; max|value replay-rollout| {vgap:.2e}; grads finite {grads_ok} ({n_grad} tensors); "
                  f"rollout {t_roll:.2f}s, replay+backward {t_upd:.2f}s, peak {peak:.1f} GiB", flush=True)
            ok &= dev < 1e-3 and grads_ok and bool(torch.isfinite(out["values"]).all())
    print("RL PATH", "OK" if ok else "FAILED", flush=True)


if __name__ == "__main__":
    main()
