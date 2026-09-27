# Plan: RL fine-tuning of the UMICore GR00T N1.7 policy with RLinf

Context (verified 2026-09-23..27): the company policy is a GR00T-N1.7-3B fine-tune with a custom
processor (rot6d, cumulative EE-local actions, 16-step chunk @15 fps, 14-dim state, two wrist
cameras), embodiment tag `umi_bimanual_v2` from `deps/umi-data-sdk`. There is no simulator and no
success detector for the UMI tasks. Reward therefore comes from offline data (STEAM advantage
labels) first, real-robot RL later.

## Phase 0 — RLinf loads the company model (no robot, no sim)
1. Reproducible environment: venv built by RLinf's installer with `GR00T_PATH` pointing at the
   pinned `umi-robot-policy` submodule, plus `umi-data-sdk`.
2. Glue in `umi_rl/`: register embodiment tag `umi_bimanual_v2`, an observation converter for the
   UMI observation dict (two wrist images, 14-dim state, prompt) and an action converter for the
   20-dim × 16-step chunk. RLinf itself is edited only where a registry cannot be extended.
3. Model config `configs/model/gr00t_n1d7_umi.yaml` (`action_dim 20`, `num_action_chunks 16`,
   `padding_value 0`).
4. Parity check: the same dataset sample through RLinf's wrapper and through the fork's own policy
   gives the same action (same flow-matching noise). Without parity, RL is meaningless.

## Phase 1 — Offline RL with STEAM advantages
Use the existing teleop datasets and their STEAM labels as the reward/advantage signal. Check how
far RLinf's offline path (`train_offline_rl.py`, STEAM value model) already fits the LeRobot data
and the fork's processor; write the dataset adapter; run a short training on 8 GPUs; evaluate
open-loop ADE on the holdout against the SFT baseline.

## Phase 2 — Real-robot RL (later)
Robot driver + task env under RLinf's real-world stack, human/keyboard or classifier reward, PPO
with the flow-SDE policy. Needs a safety operator; out of scope for now.

## Working rules
- `umi-robot-policy` is never modified in place; needed changes go to `patches/` and, when
  accepted, upstream via a `feature/COR-N/...` branch.
- Small commits, one topic each, so the development can be followed.
- GPU pinning goes through RLinf `cluster.component_placement` (physical ranks); never export
  `CUDA_VISIBLE_DEVICES` before `ray start`.
