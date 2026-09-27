#!/usr/bin/env python3
"""Pull one training batch (STEAM-stamped) from a labelled dataset and push it through the
company model's training forward, both the fork's object and RLinf's model object."""
from __future__ import annotations

import argparse
import logging
import os
import time

import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cfg", default=os.path.join(ROOT, "configs/model/gr00t_n1d7_umi.yaml"))
    ap.add_argument("--dataset", default="/data/dataset/26-W33-TELE2-rebot-batch1")
    ap.add_argument("--tag", default="umi_bimanual_v2")
    ap.add_argument("--label-version", default="v0.2.16")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
    import umi_rl.model  # noqa: F401
    from umi_rl.offline.data import OfflineDatasetSpec, build_dataset, build_loader, build_processor
    from rlinf.models import get_model

    cfg = OmegaConf.load(a.model_cfg)
    proc = build_processor(cfg.model_path)
    spec = OfflineDatasetSpec(a.dataset, a.tag, steam_optimality=True, steam_optimality_threshold=a.threshold)
    t = time.time()
    ds = build_dataset(proc, [spec], steam_label_version=a.label_version)
    loader = build_loader(ds, proc, a.batch_size)
    batch = next(iter(loader))
    print(f"dataset built + first batch in {time.time()-t:.0f}s")
    inputs = batch["inputs"] if "inputs" in batch else batch
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"  {k:18s} {tuple(v.shape)} {v.dtype}")
    opt = inputs.get("optimality")
    print(f"  optimality: {opt.tolist() if opt is not None else None}")

    model = get_model(cfg)
    model.train()
    t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model.sft_forward(batch)  # the fork's training loss through RLinf's SFT hook
    loss.backward()
    stats = {k: round(v, 4) for k, v in list(model.last_sft_stats.items())[:8]}
    print(f"sft_forward: loss={float(loss):.4f} in {time.time()-t:.1f}s  stats={stats}")
    print(f"  backward ok; peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB for B={a.batch_size}")


if __name__ == "__main__":
    main()
