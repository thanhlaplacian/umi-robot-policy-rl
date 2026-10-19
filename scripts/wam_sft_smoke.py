#!/usr/bin/env python3
"""W1 check: umi_wam's sft_forward (the fork's UmiWam.forward loss) on real batches from an LPR1 dataset through the
WAM processor in train mode. The private und cache holds only the simulator prompts, so dataset prompts go through
the live und tower, exactly as the fork's trainer does under wam_und_live.

usage: wam_sft_smoke.py [--dataset DIR] [--batch 2] [--steps 2]
"""
from __future__ import annotations

import argparse
import time

import torch

from wam_common import DEFAULT_MODEL_CFG, load_model


def wam_dataset(model_path: str, proc, dataset: str):
    """The checkpoint's own data config (WAM future-frame stamping, cumulative actions, instruction rewrite ...) with
    the mixture replaced by ``dataset`` (settings of the first mixture entry)."""
    import dataclasses

    from omegaconf import OmegaConf
    from umi_data_sdk.core.data_config import DataConfig, SingleDatasetConfig
    from umi_data_sdk.dataset.factory import DatasetFactory

    data = OmegaConf.to_container(OmegaConf.load(f"{model_path}/experiment_cfg/conf.yaml").data, resolve=True)
    ds_fields = {f.name for f in dataclasses.fields(SingleDatasetConfig)}
    entry = {k: v for k, v in data["datasets"][0].items() if k in ds_fields}
    entry.update(dataset_paths=[dataset], mix_ratio=1.0, val_dataset_path=None)
    fields = {f.name for f in dataclasses.fields(DataConfig)}
    kw = {k: v for k, v in data.items() if k in fields and k not in ("datasets", "modality_configs")}
    dropped = sorted(set(data) - fields)
    if dropped:
        print("data config keys this SDK does not know (dropped):", dropped, flush=True)
    cfg = DataConfig(datasets=[SingleDatasetConfig(**entry)], modality_configs=proc.modality_configs, **kw)
    train_ds, _ = DatasetFactory(cfg).build(proc)
    return train_ds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG)
    ap.add_argument("--dataset", default="/data/dataset/TELE/TELE-hand2-lpr1-2026-W36-batch1")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2)
    a = ap.parse_args()
    from gr00t.model.umi_wam.processing_umi_wam import UmiWamProcessor
    from gr00t.model.umi_wam.und_cache import cache_snapshot
    from gr00t.model.umi_wam.und_tower import UndTower
    from gr00t.model.umi_wam.setup import UND_LIVE_TOKENS
    from umi_rl.offline.data import build_loader

    model, cfg = load_model(a.model_cfg)
    model.set_und_tower(UndTower.from_transformer_dir(cache_snapshot(model.und_cache.meta) / "transformer"))
    proc = UmiWamProcessor.from_pretrained(cfg.model_path, wam_und_cache_dir=cfg.wam_und_cache_dir)
    proc.train()
    proc.enable_live_und(UND_LIVE_TOKENS)
    ds = wam_dataset(cfg.model_path, proc, a.dataset)
    it = iter(build_loader(ds, proc, batch_size=a.batch))
    if getattr(model.config, "wam_video_skip_cf", False):
        # the fork's mixture stamps arm_cf / task_cf rows (counterfactual prompts); one plain dataset carries neither
        print("note: wam_video_skip_cf off for this smoke (no counterfactual rows in a one-dataset mixture)", flush=True)
        model.config.wam_video_skip_cf = False
    # the training run listed its own video/action-only datasets; this one-dataset mixture contains none of them
    model.config.wam_video_skip_datasets, model.config.wam_action_skip_datasets = [], []
    model.set_video_skip_datasets([a.dataset]); model.set_action_skip_datasets([a.dataset])
    model.train(); model.set_frozen_modules_to_eval_mode()
    params = [p for p in model.parameters() if p.requires_grad]
    for s in range(a.steps):
        batch = next(it)
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        loss = model.sft_forward(batch)
        loss.backward(); torch.cuda.synchronize()
        ok = bool(torch.isfinite(loss)) and all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        print(f"step {s}: loss {float(loss):.4f}; stats {model.last_sft_stats}; finite {ok}; "
              f"{time.time()-t0:.2f}s, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    print("SFT FORWARD", "OK" if ok else "FAILED", flush=True)


if __name__ == "__main__":
    main()
