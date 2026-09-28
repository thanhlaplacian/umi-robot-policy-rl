#!/usr/bin/env python3
"""Online RL (PPO) entrypoint: RLinf's examples/embodiment/train_embodied_agent.py with the
umi_rl model/env registrations imported first. usage: bash scripts/run_rl.sh <config under configs/rl>"""
import os
import runpy
import sys

import umi_rl.model  # noqa: F401  registers gr00t_n1d7_umi in the driver before validate_cfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = os.path.join(ROOT, "third_party/rlinf/examples/embodiment/train_embodied_agent.py")
if __name__ == "__main__":
    sys.argv = [ENTRY, "--config-path", os.path.join(ROOT, "configs/rl"), *sys.argv[1:]]
    runpy.run_path(ENTRY, run_name="__main__")
