#!/usr/bin/env python
"""Smoke of GarmentRLEnv: build the jeans scene, reset N envs, settle, hold, push the grasping arm
toward the garment with scripted deltas and report contacts/reward/timings; writes frames.
  bash scripts/in_container.sh -g 0 python scripts/garment_env_smoke.py --num-envs 2
"""
import argparse, json, os, sys, time
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=2); ap.add_argument("--stack", type=int, default=1)
    ap.add_argument("--arm", default="left"); ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--env-cfg", default=None); ap.add_argument("--out", default="/home/thanh/rlinf-runs/logs/umi-rl/garment_smoke")
    a = ap.parse_args()
    from PIL import Image
    from umi_rl.envs.laplacian.garment_env import GarmentRLConfig, GarmentRLEnv
    os.makedirs(a.out, exist_ok=True)
    ecfg = json.loads(a.env_cfg) if a.env_cfg else {}
    t = time.time()
    env = GarmentRLEnv(GarmentRLConfig(stack_count=a.stack, arm=a.arm, **ecfg), num_envs=a.num_envs, device="cuda:0")
    print(f"env built in {time.time()-t:.0f}s  nq={env.model.nq} nu={env.model.nu} garments={env.garments} frame_steps={env.frame_steps} prompt={env.cfg.prompt!r}")
    t = time.time(); obs = env.reset(seeds=list(range(a.num_envs))); print(f"reset+settle {time.time()-t:.1f}s")
    for k, v in obs.items(): print(f"  obs[{k}] {tuple(v.shape)} {v.dtype}")
    pts = env.garment_points(); c = pts.mean(1)
    tcp = env.p.body_pos[:, env.tip_ids[env.k_active]]
    print(f"garment centroid {c.cpu().numpy().round(3).tolist()}  shelf_top {env.shelf_top:.3f}  bottom z {pts[...,2].amin(-1).cpu().numpy().round(3).tolist()}")
    print(f"active TCP {tcp.cpu().numpy().round(3).tolist()}  d={env.min_tcp_object_distance().cpu().numpy().round(3).tolist()}")
    print(f"state[0] {obs['states'][0].cpu().numpy().round(3).tolist()}")
    for name, im in [(k, v) for k, v in obs.items() if k.startswith("cam")]:
        Image.fromarray(im[0].cpu().numpy()).save(os.path.join(a.out, f"{name}_reset.png"))
    # hold for 5 steps (zero deltas, gripper 0.5 = half), then move toward the garment in base frame
    R, pos, Rb, pb = env.ee_poses()
    k = env.k_active
    goal_w = c.clone()  # centroid world
    for i in range(a.steps):
        act = torch.zeros(a.num_envs, 14, device=env.device)
        if i >= 5:
            R, pos, Rb, pb = env.ee_poses()
            tcp_w = env.p.body_pos[:, env.tip_ids[k]]
            dw = goal_w - tcp_w
            dw = dw / (torch.linalg.vector_norm(dw, dim=-1, keepdim=True) + 1e-9) * 0.01  # 1 cm per step toward the centroid
            d_local = torch.einsum("nji,nj->ni", Rb[:, None].expand(-1, 1, -1, -1)[:, 0] @ R[:, k], dw)  # world -> EE-local
            o = 0 if k == 0 else 7
            act[:, o + 1 : o + 4] = d_local
        act[:, 0] = act[:, 7] = 0.4  # gripper command
        t = time.time(); obs, r, term, trunc, info = env.step(act); dt = time.time() - t
        pads, lower = env.pad_contacts()
        print(f"step {i:2d} {dt:.2f}s {env.timing}  r={r.cpu().numpy().round(2).tolist()} d={env.min_tcp_object_distance().cpu().numpy().round(3).tolist()} pads={pads.int().cpu().numpy().tolist()} rise={env.last_reward_terms['rise'].cpu().numpy().round(3).tolist()} term={term.cpu().numpy().tolist()}")
    for name, im in [(k2, v) for k2, v in obs.items() if k2.startswith("cam")]:
        Image.fromarray(im[0].cpu().numpy()).save(os.path.join(a.out, f"{name}_end.png"))
    print("frames in", a.out)


if __name__ == "__main__":
    main()
