# Phase 2: a simulator environment for the policy (laplacian-gym in RLinf)

Status (2026-09-28): the batched env works, the company policy runs closed-loop in it, and the
full RLinf PPO loop (env workers -> rollout -> actor FSDP update -> weight sync) completed a
2-epoch smoke. Branch `feature/laplacian-gym-env`.

## Decision

`laplacian-gym` (the company's CUDA simulator: MuJoCo Warp physics, gsplat background, nvdiffrast
meshes, LPR1) is used **in-process** as an RLinf env backend, in the same venv as RLinf and the
policy fork. Its `torch==2.14` pin turned out to be a measured stack, not an API requirement
(`pyproject` says `torch>=2.7`, bootstrap checks only CUDA 13.0): the core installs and runs
on torch 2.11+cu130 and H100 (`scripts/setup_gym.sh`). Left out on purpose: cuRobo and the
JAX/PyRoki planner (RL needs per-step IK, not grasp planning), the LeRobot writer, viewer and
annotation extras. gsplat is kept: it is the only source of the photoreal room the policy was
trained on (the scene exists only as a 3D Gaussian splat plus an untextured collision mesh).

## What was built

| Piece | Where | Notes |
|---|---|---|
| Gym submodule pinned `6ef7231`, LFS assets 3.1 GB | `third_party/laplacian-gym` | never edited; the mujoco-warp CCD patch is applied to the venv by `setup_gym.sh` |
| `LaplacianRLEnv` (batched) | `umi_rl/envs/laplacian/rl_env.py` | bypasses the gym's planner task: samples objects on the shelf with the generator's bands, base at the nominal pose, settles 0.5 s; `step([N,14])` integrates the policy's EE-local delta on the TCP pose, solves joint targets with damped-least-squares IK on MuJoCo Jacobians (CPU, 4 iters), gripper `ctrl = -1.0372 * g`, 40 physics substeps per 15 Hz frame; wrist cameras rendered at 640x480 by scaling the imported fisheye calibration, head at native 640x400; success = every object inside the generator's basket AABB |
| RLinf backend | `third_party/rlinf/rlinf/envs/sim/laplacian_gym/laplacian_env.py`, `SupportedEnvType.LAPLACIAN_GYM`, pass-through in `prepare_actions` | mirrors `maniskill_env.py` (group reset ids, `chunk_step`, auto-reset infos); obs = `states [N,14]`, `wrist_images [N,2,480,640,3]`, `main_images = cam_head` |
| Configs | `configs/env/laplacian_pick_place.yaml`, `configs/rl/laplacian_ppo_gr00t_umi_smoke.yaml` | env 0-1, rollout 2-3, actor 4-7 |
| Tools | `scripts/setup_gym.sh`, `scripts/in_container.sh`, `scripts/gym_env_smoke.py`, `scripts/run_rl.sh` | all builds and tests run inside the SFT container |

## Measured (container `umi-thanh-maskfix`, GPU 7, 2 envs, `scripts/gym_env_smoke.py`)

| Item | Value |
|---|---|
| First-run JIT (warp + gsplat + nvdiffrast) | ~10 min once; cached afterwards |
| Env build / reset | 50 s / 6.6 s (includes CUDA graph capture on the first step) |
| Step, 2 envs, 3 cameras | 0.35-0.48 s, of which render 0.35 s, physics 2 ms, IK 3 ms |
| Hold test (zero deltas, 5 steps) | TCP drift 0.0000 m |
| Scripted +2 cm x5 along the EE x axis, then back | TCP moved ~6 cm and returned to within 1 cm (servo lag; fine for RL) |
| Policy closed-loop, 3 chunks | policy 0.2-0.6 s/chunk; TCP-to-nearest-object distance 0.144 -> 0.095 -> 0.062 m |
| Renders | `/home/thanh/rlinf-runs/logs/umi-rl/gym_smoke/*.png` (fisheye wrists with grippers, basket, objects; pinhole head) |

## RLinf PPO smoke (`configs/rl/laplacian_ppo_gr00t_umi_smoke.yaml`, run `laplacian_ppo_gr00t_umi_smoke`)

Placement env 0-1 (2 workers x 4 envs), rollout 2-3, actor 4-7 (FSDP), 96 steps per rollout
epoch (6 chunks of 16), 2 epochs, wandb project `finetune-gr00t-n1d7`.

| Item | Value |
|---|---|
| Rollout epoch (8 envs x 96 steps, 3 cameras) | 66-70 s (`env/env_interact_step` 64-67 s; render-bound) |
| Trajectories per epoch | 8, episode_len 96, success 0 (sparse reward, 6.4 s episodes: expected) |
| Actor update | 4.0-4.6 s per epoch; ratio 1.01-1.02, clip fraction 0.04, approx_kl -0.006 / -0.016 |
| Losses | policy 0.022 -> 4e-4, value 90.5 -> 10.2 (value head starts from scratch), grad_norm 2770 -> 1280 |
| Weight sync actor -> rollout | 6.4 s / 2.0 s |
| Whole run incl. env build and model loads | ~4 min |

Fixes that were needed on the RLinf side (all in the subtree, small): a lazy `.pth` hook so Ray
workers see `model_type gr00t_n1d7_umi` (`umi_rl/autoregister.py`), the new type added to the
GR00T N1.7 special cases (`prev_logprobs`, `clip_ratio_c`, rollout batching), and an
action pass-through branch. The env backend itself needed no change after the standalone smoke.

## Reward (`RewardConfig` in `umi_rl/envs/laplacian/rl_env.py`)

Per-term flags, all off by default (legacy sparse + optional reach term). `staged: true` enables
the PlaceSphere-style stage overwrite on the active object (reach 2·φ → grasp 4+φ(goal) → lift
6+φ(goal) → in basket 8 + release_bonus·opening) plus 10 per object already in the basket and
`success` (13) on completion; `idle_arm_penalty`, `disturb_penalty` and `regress_penalty` add the
DOSW1 idle-arm term, the digital-twin disturbance term and a latched −w when an object leaves the
basket. Grasp = finger-geom contact with the object (from `physics.contacts()`) while the gripper
is commanded closed beyond 0.3; lift = grasped and ≥ 2.5 cm above the reset height. Formulas and
the catalog they come from: the "Pick-and-Place Reward Catalog" artifact.

Check with the SFT policy (v0.16.9, 240 steps, `--reward '{"staged":true,"idle_arm_penalty":0.1,"disturb_penalty":1.0,"regress_penalty":2.0}'`):
seed 2 reached the grasp stage for 12 steps and the lift stage for 5 steps (max stage reward 6.02),
seed 4 the grasp stage for 8 steps; cumulative reward +264 / +259, penalties ≈ 0 (arms near home,
other object untouched).

## PPO fine-tuning attempts (2026-09-28/29)

| Run | Setup | Outcome |
|---|---|---|
| v1 `laplacian_ppo_v0169_staged` | 2 objects, 24 envs x 240 steps, lr 5e-6, kl 0.01, staged reward | first 11 epochs had negative returns: RLinf's `action_noise_scale 0.1` adds N(0, 0.1) to the DECODED per-step EE deltas (metres), the arms flailed and knocked objects off the shelf. Fixed (`action_noise_scale 0`, fall = failure); 27 epochs flat afterwards (grasp 1-3 %, one object in the basket in 3 epochs, never both) |
| v2 `laplacian_ppo_v0169_staged_1obj_v2` | 1 object, lr 2e-5, kl 0.001, 2 update epochs, 120 epochs (~10 h, 8 GPUs) | return 18.4 -> 20.0 (20-epoch block means), grasp 1.2 -> 1.7 % of steps, lift 0.2 -> 0.5 %; success 0 in epochs 1-20, then 13 successful episodes out of ~2,600 in epochs 21-120 (0.4-0.7 % per block), no sustained increase. Curves: `/home/thanh/rlinf-runs/logs/umi-rl/ppo_v2_curves.png` |

Held-out check, 8 seeds, one object, eval-mode policy (`scripts/gym_episode_video.py --ckpt ...`):

| Policy | success | seeds with grasp contact | seeds with lift | grasp steps (sum) | mean cumulative reward |
|---|---|---|---|---|---|
| v0.16.9 base | 0/8 | 5 | 2 | 71 | 201 |
| PPO v2 step 120 | 0/8 | 7 | 1 | 135 | 250 |

Reading: PPO learned to reach the object and hold finger contact with a closed gripper (the
stage-4 reward) more often, but not to get the object between the fingers before closing, so
lifts did not increase. The zero-shot policy closes the gripper ~8 cm early in this simulator
(depth misjudged from the sim fisheye), and PPO from a ~0 % base rate with 360 chunk samples per
epoch does not find the corrected grasp. Next steps in order of expected value: warm-start with
the gym's planner demonstrations (`laplacian-gym generate`, same scene, cameras and LeRobot
`umi_bimanual` format), then PPO; add a finger-alignment shaping term (object between the fingers
before closing) and gate the grasp stage on it; consider GRPO with fixed reset ids so the rare
successes are contrasted within a group.

## Which company checkpoint to start from (sim, 8 seeds, one object, eval mode, 2026-09-29)

| Checkpoint | success | seeds w/ grasp | seeds w/ lift | grasp steps | lift steps | mean min dist (m) | mean cum. reward |
|---|---|---|---|---|---|---|---|
| v0.16.10 @ 50000 | **1/8** | 4 | 3 | 113 | 45 | 0.097 | 199 |
| v0.16.13 @ 60000 | 0/8 | 4 | 2 | 70 | 22 | 0.107 | 172 |
| v0.16.9 @ 3000 | 0/8 | 5 | 2 | 71 | 16 | 0.067 | 201 |
| v0.15.5 @ 20000 | 0/8 | 0 | 0 | 0 | 0 | 0.224 | 73 |
| PPO v2 step 120 (from v0.16.9) | 0/8 | 7 | 1 | 135 | 7 | 0.074 | 250 |

v0.16.10@50000 is the only checkpoint that completed the task zero-shot in this protocol and has
the most lift steps; it is the recommended base for further RL / warm-start. v0.17.x checkpoints
(arm-active conditioning, COR-200) load under the pinned fork but drop 8 `arm_active_*` tensors,
so evaluating them faithfully needs the submodule bumped to a post-COR-200 commit.

## Known gaps

- Throughput is render-bound (~3 policy steps/s for 2 envs). Batched rendering of more envs is
  the lever to measure next; otherwise several env workers per GPU.
- Reward is sparse (basket AABB); no lift bonus or dense shaping yet (`dense_reward` knob exists).
- Base pose and lighting are not randomized; the gym supports both.
- Continuous gripper commands are untested by the gym's own data (demos use open/closed only).
- `numpy` was upgraded to 2.4 in the venv (gym requires `>=2`); only `rlinf-libero` objects.
