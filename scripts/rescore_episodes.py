#!/usr/bin/env python
"""Re-score logged pick-and-place episodes offline with the strict placement rule.

The old success fired when an object's centre entered the generator's loose basket box (often while
still carried over the rim). The strict rule: centre inside the footprint shrunk by ``xy_margin`` and
below the rim, gripper not grasping (no grasp term; the logs have no contact flags, so the released
test uses the acting gripper command < ``grip_open``), held for ``hold`` consecutive steps. Episodes
that were terminated early by the old rule are scored on their logged steps only, so a strict success
needs the hold to have completed before the old termination.
  python scripts/rescore_episodes.py --tags base_gym9e3_boost,v8b_100_gym9e3_boost
"""
import argparse, glob, json, os, re
from collections import defaultdict
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from umi_rl.envs.laplacian.rl_env import BASKET_X, BASKET_Y, BASKET_Z_BELOW_SHELF
SHELF_TOP, BASKET_RIM = 1.172265, 1.337
D = "/home/thanh/rlinf-runs/logs/umi-rl/gym_episode"


def strict(log, xy_margin, below_rim, hold, grip_open):
    run = 0
    for x in log:
        ok = True
        for o in x["objects_xyz"]:
            ok &= (BASKET_X[0] + xy_margin < o[0] < BASKET_X[1] - xy_margin and BASKET_Y[0] + xy_margin < o[1] < BASKET_Y[1] - xy_margin
                   and SHELF_TOP - BASKET_Z_BELOW_SHELF < o[2] < BASKET_RIM - below_rim)
        ok &= max(x["grip"]) < grip_open
        run = run + 1 if ok else 0
        if run >= hold:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True); ap.add_argument("--xy-margin", type=float, default=0.03); ap.add_argument("--below-rim", type=float, default=0.0)
    ap.add_argument("--hold", type=int, default=5); ap.add_argument("--grip-open", type=float, default=0.3)
    a = ap.parse_args()
    for tag in a.tags.split(","):
        per = defaultdict(lambda: [0, 0, 0])  # object -> [n, old, strict]
        for f in sorted(glob.glob(f"{D}/episode_{tag}_*_seed[0-9].json")):
            m = re.match(rf"episode_{re.escape(tag)}_(.+)_seed(\d+)\.json", os.path.basename(f))
            obj = m.group(1); d = json.load(open(f))
            c = per[obj]; c[0] += 1; c[1] += int(d["success"]); c[2] += int(strict(d["log"], a.xy_margin, a.below_rim, a.hold, a.grip_open))
        tot = [sum(v[i] for v in per.values()) for i in range(3)]
        cells = " | ".join(f"{o}: old {v[1]}/{v[0]} strict {v[2]}/{v[0]}" for o, v in sorted(per.items()))
        print(f"{tag:26s} | {cells} | all: old {tot[1]}/{tot[0]} strict {tot[2]}/{tot[0]}")


if __name__ == "__main__":
    main()
