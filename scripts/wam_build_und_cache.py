#!/usr/bin/env python3
"""Build a private und K/V cache for a list of prompts (the shared caches under /data/models/wam_und_cache are mode
600, owner ubuntu). Rows are keyed through the processor's own path (``_formalize`` -> caption -> tokenize ->
``prompt_key``), exactly as serving looks them up, and computed by the und tower of the Cosmos3-Edge snapshot the
checkpoint names (readable). The cache meta carries the checkpoint's caption settings.

usage: wam_build_und_cache.py --out DIR [--ckpt DIR] [--prompt TEXT ...]
default prompts: the three simulator prompts (rigid pick-place, garment left / right)
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch

SIM_PROMPTS = [
    "Pick up the objects on the shelf and place them in the basket",
    "Pick up the garment on the shelf. the left arm grasps and the right arm does nothing",
    "Pick up the garment on the shelf. the left arm does nothing and the right arm grasps",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/data/models/UMICore-v0.18.39/checkpoint-3000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", action="append", default=None, help="repeatable; default: the simulator prompts")
    ap.add_argument("--force", action="store_true", help="replace an existing cache at --out")
    a = ap.parse_args()
    prompts = a.prompt or SIM_PROMPTS
    from gr00t.configs.model.umi_wam import UmiWamConfig
    from gr00t.model.umi_wam.processing_umi_wam import UmiWamProcessor
    from gr00t.model.umi_wam.und_cache import UndKVCache, build_und_kv, cache_snapshot, prompt_key
    from gr00t.model.umi_wam.und_tower import UndTower
    from gr00t.model.umi_wam.umi_wam import UmiWam

    out = Path(a.out)
    if out.exists():
        if not a.force:
            raise SystemExit(f"{out} exists; pass --force to replace it")
        shutil.rmtree(out)
    cfg = UmiWamConfig.from_pretrained(a.ckpt)
    snapshot = cache_snapshot({"snapshot": cfg.wam_ckpt_dir})
    caption = {"view_point": cfg.wam_view_point, "future_frames": cfg.wam_future_frames, "fps": cfg.wam_fps,
               "obs_hw": list(cfg.wam_obs_hw)}
    proc0 = UmiWamProcessor.from_pretrained(a.ckpt)  # prompt_ids works without a readable cache
    formal = [proc0._formalize(p) for p in prompts]
    ids = [proc0.prompt_ids(f) for f in formal]
    for p, f, i in zip(prompts, formal, ids):
        print(f"{len(i):4d} tokens key {prompt_key(i)[:10]}  {f!r}" + ("" if f == p else "  (formalized)"), flush=True)
    t0 = time.time()
    tower = UndTower.from_transformer_dir(snapshot / "transformer").to(torch.bfloat16).cuda()
    entries = [(i, *[t.cpu() for t in build_und_kv(tower, i)]) for i in ids]
    meta = {"config": "umi-robot-policy-rl scripts/wam_build_und_cache.py", "snapshot": str(snapshot),
            "sentences": len(prompts), "closure_sha1": "", "caption": caption}
    UndKVCache.from_entries(entries, meta).save(out)
    print(f"cache with {len(entries)} rows built in {time.time()-t0:.0f}s -> {out}", flush=True)
    del tower
    torch.cuda.empty_cache()
    model = UmiWam.from_pretrained(a.ckpt, wam_und_cache_dir=str(out), dtype=torch.bfloat16)
    proc = UmiWamProcessor.from_pretrained(a.ckpt, wam_und_cache_dir=str(out))
    for p in prompts:
        assert proc.has_sentence(p), p
    print(f"verified: the checkpoint loads with the cache ({len(model.und_cache)} rows) and serves every prompt", flush=True)


if __name__ == "__main__":
    main()
