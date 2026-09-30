#!/usr/bin/env python
"""Balanced per-object evaluation launcher: one gym_episode_video.py episode at a time per GPU.

Generates per-GPU queue scripts under /home/thanh/rlinf-runs/tools/q and starts them detached inside
the container (scripts/in_container.sh). Results land in gym_episode/episode_<tag>_<object>_seed<s>.*
  python scripts/eval_queue.py --ckpt /path --tag base_boost --objects ROB-001,ROB-003,RTC-001 --seeds 0-7 --gpus 0-7
Summarise with: python scripts/eval_queue.py --summary <tag>[,<tag>...]
"""
import argparse, json, os, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDIR = "/home/thanh/rlinf-runs/tools/q"
EPD = "/home/thanh/rlinf-runs/logs/umi-rl/gym_episode"
REWARD_V5 = {"staged": True, "idle_arm_penalty": 0.05, "disturb_penalty": 0.5, "regress_penalty": 2.0, "align_dist": 0.05, "align_bonus": 1.0,
             "gate_grasp_on_align": True, "align_needs_open": False, "close_bonus": 1.0}


def rng(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def summary(tags, objects, seeds):
    print(f"{'tag':28s} | " + " | ".join(f"{o:>14s}" for o in objects) + " |            all")
    for tag in tags:
        cells, S, L, N = [], 0, 0, 0
        for o in objects:
            s = l = n = 0
            for sd in seeds:
                p = f"{EPD}/episode_{tag}_{o}_seed{sd}.json"
                if not os.path.exists(p): continue
                d = json.load(open(p)); n += 1; s += int(d["success"]); l += int(any(x["terms"].get("lifted", 0) for x in d["log"]))
            cells.append(f"{s}/{n} (lift {l})"); S += s; L += l; N += n
        print(f"{tag:28s} | " + " | ".join(f"{c:>14s}" for c in cells) + f" | {S}/{N} (lift {L})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--tag"); ap.add_argument("--objects", default="ROB-001,ROB-003,RTC-001")
    ap.add_argument("--seeds", default="0-7"); ap.add_argument("--gpus", default="0-7")
    ap.add_argument("--env-cfg", default='{"spawn_count":1,"gripper_mode":"boost","gripper_thresh":0.3}')
    ap.add_argument("--reward", default=json.dumps(REWARD_V5)); ap.add_argument("--extra", default="", help="extra gym_episode_video.py args")
    ap.add_argument("--summary", default=None, help="comma-separated tags to summarise instead of launching")
    ap.add_argument("--task", default="pick_place", choices=["pick_place", "garment"], help="garment: one 'object' (the garment), no object_types, --task garment forwarded")
    a = ap.parse_args()
    objects, seeds, gpus = a.objects.split(","), rng(a.seeds), rng(a.gpus)
    if a.task == "garment":
        objects = ["garment"]
        if a.reward == json.dumps(REWARD_V5):
            a.reward = "{}"
        a.extra = (a.extra + " --task garment").strip()
    if a.summary:
        return summary(a.summary.split(","), objects, seeds)
    jobs = [(o, s) for o in objects for s in seeds]
    os.makedirs(QDIR, exist_ok=True)
    base = json.loads(a.env_cfg)
    for gi, g in enumerate(gpus):
        mine = jobs[gi::len(gpus)]
        lines = ["#!/bin/bash"]
        for o, s in mine:
            ecfg = json.dumps(base if a.task == "garment" else {**base, "object_types": [o]})
            log = f"/home/thanh/rlinf-runs/logs/lgym-ep-{a.tag}-{o}-s{s}.log"
            lines.append(f"timeout 3000 python scripts/gym_episode_video.py --seed {s} --tag {a.tag}_{o}_seed --ckpt {a.ckpt} --reward '{a.reward}' --env-cfg '{ecfg}' {a.extra} > {log} 2>&1 < /dev/null")
        qf = f"{QDIR}/{a.tag}_gpu{g}.sh"
        open(qf, "w").write("\n".join(lines) + "\n"); os.chmod(qf, 0o755)
        subprocess.run(["bash", f"{ROOT}/scripts/in_container.sh", "-g", str(g), "bash", "-c", f"setsid nohup {qf} > /dev/null 2>&1 < /dev/null &"], check=True)
        print(f"gpu {g}: {len(mine)} episodes queued ({qf})")


if __name__ == "__main__":
    main()
