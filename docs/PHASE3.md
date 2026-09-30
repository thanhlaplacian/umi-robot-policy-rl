# Phase 3 (overnight 2026-09-30 -> 10-01): new gym physics, garment task, v0.17.9

Five requests: (1) re-evaluate the base v0.16.10 on the new gripper physics, (2) RL-finetune v0.16.10 with
videos, (3) build a garment-pick env, (4) pick a company IL checkpoint for it and evaluate with videos,
(5) RL-finetune that checkpoint and compare. Videos: `/home/thanh/rlinf-runs/logs/umi-rl/gym_episode/`
(`episode_<tag>_<object>_seed<s>.mp4` + `.json` + `_reward.png`).

## 0. Two integration findings that change everything before it

**RLinf's train-mode sampler ran the DiT without the fork's `cond_add`** (embodiment row, progress-conditioning
row, and on v0.17 the arm-active rows). Upstream N1.7 has no such slot, so RLinf's `sample_mean_var_val`
never passes one; the fork's `get_action` (our eval path) does. Every PPO run before today optimised an
unconditioned DiT and was evaluated with the conditioned one. Fixed in commit c4aa22cf by a `rl_cond_add`
hook on the RL head that rebuilds the addend exactly like `get_action_with_features`.
`scripts/cond_parity.py` (same observation, same initial noise, noise level 0):

| checkpoint | max abs diff eval vs train-mode sampler, before | after |
|---|---|---|
| v0.16.10@50000 | 2.3e-2 | 3.2e-3 (bf16 sampler numerics) |
| v0.17.9@30000 | n/a | 3.9e-3 |

**Fork bump.** `third_party/umi-robot-policy` a379960 -> 9820e2d (+ SDK 1afa968) so v0.17.x checkpoints
(arm-active head + conditioning, COR-200/207) load completely: v0.17.9@30000 initialises only the RL value
head. The VLSA mask patch no longer applies and is now optional (it changed actions by <= 1 bf16 ulp).

## 1. Base v0.16.10 on gym 9e3220f (rubber gripper pads, friction 2.0, opening -4 mm)

Balanced protocol, 8 seeds x 3 objects, `gripper_mode: boost`, eval mode, one episode per GPU.

| policy | ROB-001 | ROB-003 | RTC-001 | all |
|---|---|---|---|---|
| v0.16.10@50000 + boost (old gym 6ef7231) | 2/8 | 1/8 | 2/8 | 5/24 (v5@110 numbers; base itself was 2/8 on RTC seeds) |
| v0.16.10@50000 + boost (new gym) | 0/8 | 0/8 (1 lift) | 1/8 | 1/24 |
| v5@110 + boost (new gym) | see below | | | |

tag                          |        ROB-001 |        ROB-003 |        RTC-001 |            all
base_gym9e3_boost            |   0/8 (lift 0) |   0/8 (lift 1) |   1/8 (lift 0) | 1/24 (lift 1)
v5_110_gym9e3_boost          |   0/8 (lift 1) |   1/8 (lift 3) |   0/8 (lift 3) | 1/24 (lift 7)

