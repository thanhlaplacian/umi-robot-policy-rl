#!/usr/bin/env python3
"""Which datasets of a UMICore mix carry STEAM advantage labels, and how the labels look.

usage: steam_label_audit.py [--config configs/UMICore/v0.16/v0.16.9.yaml] [--label-version v0.2.16] [--threshold 0.0]
Prints per-dataset: episodes, frames, label coverage, advantage quantiles, fraction of frames with
A >= threshold (the fork's `steam_optimality` cut), and the mix-weighted totals.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
import yaml

ROOT = "/data/users/thanh/1.2026/umi-robot-policy-rl/third_party/umi-robot-policy"


def load_cfg(p):
    d = yaml.safe_load(open(p)) or {}
    ext = d.pop("extends", None)
    if ext:
        base = load_cfg(os.path.normpath(os.path.join(os.path.dirname(p), ext))); base.update(d); d = base
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/UMICore/v0.16/v0.16.9.yaml")
    ap.add_argument("--label-version", default="v0.2.16")
    ap.add_argument("--threshold", type=float, default=0.0)
    a = ap.parse_args()
    os.chdir(ROOT)
    cfg = load_cfg(a.config)
    paths = cfg["dataset_paths"]; ratios = cfg.get("mix_ratios") or [1.0] * len(paths); tags = cfg.get("embodiment_tags") or ["?"] * len(paths)
    rows = []; tot = collections.Counter()
    for p, r, tag in zip(paths, ratios, tags):
        if not os.path.isdir(p) and p.startswith("data/"):
            p = "/" + p  # the recipes address datasets as data/... relative to a checkout that symlinks /data
        name = p.rstrip("/").split("/")[-1]
        eps = [json.loads(l) for l in open(os.path.join(p, "meta/episodes.jsonl"))] if os.path.exists(os.path.join(p, "meta/episodes.jsonl")) else []
        frames = sum(e.get("length", 0) for e in eps)
        f = os.path.join(p, f"meta/steam_advantage_{a.label_version}.npz")
        row = {"name": name, "tag": tag, "ratio": r, "episodes": len(eps), "frames": frames, "labels": None}
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            ep_keys = [k for k in z.files if k.startswith("ep_") and not k.endswith("_members")]
            adv = np.concatenate([z[k] for k in ep_keys]) if ep_keys else np.zeros(0)
            row["labels"] = {"episodes": len(ep_keys), "frames": int(adv.size), "q10": float(np.quantile(adv, .1)), "q50": float(np.quantile(adv, .5)),
                             "q90": float(np.quantile(adv, .9)), "frac_ge_thr": float((adv >= a.threshold).mean())}
            tot["labelled_frames"] += adv.size; tot["labelled_ratio"] += r; tot["pos_frames"] += int((adv >= a.threshold).sum())
        tot["frames"] += frames; tot["ratio"] += r
        rows.append(row)
    print(f"mix {a.config}: {len(rows)} datasets, labels {a.label_version}, threshold {a.threshold}\n")
    print(f"{'dataset':34s} {'tag':16s} {'ratio':>6s} {'eps':>6s} {'frames':>8s} {'lab.eps':>7s} {'lab.frames':>10s} {'q10':>6s} {'q50':>6s} {'q90':>6s} {'A>=thr':>7s}")
    for r in rows:
        L = r["labels"]
        if L: print(f"{r['name']:34s} {r['tag']:16s} {r['ratio']:6.3f} {r['episodes']:6d} {r['frames']:8d} {L['episodes']:7d} {L['frames']:10d} {L['q10']:6.2f} {L['q50']:6.2f} {L['q90']:6.2f} {L['frac_ge_thr']:7.1%}")
        else:   print(f"{r['name']:34s} {r['tag']:16s} {r['ratio']:6.3f} {r['episodes']:6d} {r['frames']:8d} {'-':>7s} {'-':>10s}")
    n_lab = sum(1 for r in rows if r["labels"])
    print(f"\nlabelled datasets: {n_lab}/{len(rows)}   labelled frames: {tot['labelled_frames']}/{tot['frames']} ({tot['labelled_frames']/max(tot['frames'],1):.1%})   "
          f"mix weight with labels: {tot['labelled_ratio']/max(tot['ratio'],1e-9):.1%}   frames A>=thr among labelled: {tot['pos_frames']/max(tot['labelled_frames'],1):.1%}")


if __name__ == "__main__":
    main()
