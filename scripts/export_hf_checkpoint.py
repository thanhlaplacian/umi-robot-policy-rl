#!/usr/bin/env python3
"""Turn an RLinf SFT checkpoint into a fork-style HF checkpoint dir the company tools can load
(Gr00tPolicy, open_loop_eval_eef.py, planner-vla export).

Reads ``<run>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt`` (written when
``fsdp_config.save_full_model_weights: true``), drops RLinf-only tensors (value head), writes
``model*.safetensors`` + index, and copies config.json / processor files / experiment_cfg from
the base checkpoint the run started from.

usage: export_hf_checkpoint.py --step-dir <.../global_step_N> --base <base ckpt dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-dir", required=True)
    ap.add_argument("--base", required=True, help="checkpoint dir the run was initialised from (config + processor files)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-gb", type=float, default=4.5)
    a = ap.parse_args()
    from safetensors.torch import save_file

    fw = os.path.join(a.step_dir, "actor", "model_state_dict", "full_weights.pt")
    if not os.path.exists(fw):
        raise SystemExit(f"{fw} not found: run with fsdp_config.save_full_model_weights: true")
    sd = torch.load(fw, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict) or not sd:
        raise SystemExit(f"unexpected content in {fw}: {type(sd)}")
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    drop = [k for k in sd if k.startswith("action_head.value_head") or ".value_head." in k]
    for k in drop:
        sd.pop(k)
    # FSDP/DCP may leave a wrapper prefix; strip a leading "_fsdp_wrapped_module." / "module."
    def clean(k):
        for pre in ("_fsdp_wrapped_module.", "module."):
            k = k.replace(pre, "")
        return k
    sd = {clean(k): (v.contiguous() if torch.is_tensor(v) else v) for k, v in sd.items()}
    # the fork stores bf16 checkpoints; keep the dtype of the base checkpoint
    base_index = os.path.join(a.base, "model.safetensors.index.json")
    base_dtype = None
    if os.path.exists(base_index):
        from safetensors import safe_open
        first = sorted(set(json.load(open(base_index))["weight_map"].values()))[0]
        with safe_open(os.path.join(a.base, first), "pt") as f:
            base_dtype = f.get_tensor(next(iter(f.keys()))).dtype
    if base_dtype is not None:
        sd = {k: (v.to(base_dtype) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sd.items()}

    os.makedirs(a.out, exist_ok=True)
    shards, cur, cur_bytes, limit = [], {}, 0, int(a.shard_gb * 2**30)
    for k in sorted(sd):
        n = sd[k].numel() * sd[k].element_size()
        if cur and cur_bytes + n > limit:
            shards.append(cur); cur, cur_bytes = {}, 0
        cur[k] = sd[k]; cur_bytes += n
    if cur:
        shards.append(cur)
    weight_map = {}
    for i, sh in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, os.path.join(a.out, name), metadata={"format": "pt"})
        weight_map.update({k: name for k in sh})
    total = sum(v.numel() * v.element_size() for v in sd.values())
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    for f in ("config.json", "processor_config.json", "statistics.json", "embodiment_id.json", "training_processor_config.json", "wandb_config.json"):
        src = os.path.join(a.base, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out, f))
    if os.path.isdir(os.path.join(a.base, "experiment_cfg")):
        shutil.copytree(os.path.join(a.base, "experiment_cfg"), os.path.join(a.out, "experiment_cfg"), dirs_exist_ok=True)
    json.dump({"rlinf_step_dir": os.path.abspath(a.step_dir), "base": os.path.abspath(a.base), "dropped_keys": drop},
              open(os.path.join(a.out, "rlinf_export.json"), "w"), indent=1)
    print(f"wrote {len(sd)} tensors ({total/2**30:.2f} GiB) in {len(shards)} shard(s) to {a.out}; dropped {len(drop)} RLinf-only keys")


if __name__ == "__main__":
    main()
