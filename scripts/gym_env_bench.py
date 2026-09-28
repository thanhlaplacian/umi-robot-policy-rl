#!/usr/bin/env python3
"""Throughput/memory of LaplacianRLEnv vs number of envs on one GPU (zero actions, full render)."""
import argparse, json, time, torch

ap = argparse.ArgumentParser(); ap.add_argument("--num-envs", type=int, required=True); ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--out", default="/home/thanh/rlinf-runs/logs/umi-rl/gym_bench"); a = ap.parse_args()
import os; os.makedirs(a.out, exist_ok=True)
from umi_rl.envs.laplacian.rl_env import LaplacianRLConfig, LaplacianRLEnv
t = time.time(); env = LaplacianRLEnv(LaplacianRLConfig(seed=0), num_envs=a.num_envs, device="cuda:0"); build = time.time() - t
t = time.time(); env.reset(); torch.cuda.synchronize(); reset_s = time.time() - t
zero = torch.zeros(a.num_envs, 14, device="cuda:0")
env.step(zero); torch.cuda.synchronize()  # warm-up (CUDA graph capture)
tim = {"ik": 0, "physics": 0, "observe": 0}; t = time.time()
for _ in range(a.steps):
    env.step(zero); torch.cuda.synchronize()
    for k in tim: tim[k] += env.timing[k]
wall = (time.time() - t) / a.steps
r = {"num_envs": a.num_envs, "build_s": round(build, 1), "reset_s": round(reset_s, 2), "step_s": round(wall, 4), "env_steps_per_s": round(a.num_envs / wall, 2),
     "ik_s": round(tim["ik"] / a.steps, 4), "physics_s": round(tim["physics"] / a.steps, 4), "render_s": round(tim["observe"] / a.steps, 4),
     "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2)}
print(json.dumps(r)); json.dump(r, open(f"{a.out}/n{a.num_envs}.json", "w"))
