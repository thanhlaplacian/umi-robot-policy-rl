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


### Success-rule correction (2026-10-01)

The user noticed in the videos that episodes ended before the object came to rest. The success test was
the generator's loose acceptance box (centre inside the footprint, height anywhere up to 8 cm above
the rim), evaluated live, so it fired while the object was still carried over the rim and terminated
the episode. Fixed in commit 0a847bfe: success now requires every object inside the basket volume
(footprint shrunk by 3 cm, centre below the rim), no finger contact, for 5 consecutive steps; the loose
box still drives the shaping stages. The basket box is `BASKET_X/BASKET_Y/BASKET_Z_*` constants.
Eval-mode episodes are not replayable (the flow's initial noise is unseeded), so the table was re-run
rather than re-scored:

| policy (strict rule) | ROB-001 | ROB-003 | RTC-001 | all |
|---|---|---|---|---|
| v0.16.10@50000 (base) | 1/8 | 0/8 | 2/8 | 3/24 |
| v8b@100 | 0/8 | 1/8 | 2/8 | 3/24 |

Under the corrected rule the RL fine-tune shows no gain over the base on the new physics (the earlier
5/24 vs 1/24 was mostly "carried over the rim" events plus sampling noise; 8 seeds per cell is +-2).
Strict-success videos: `episode_base_strict_*` and `episode_v8b_100_strict_*` with `"success": true`
in their JSON. All numbers in sections 1-2 above use the old rule.

## 3. Garment-on-shelf env (`umi_rl/envs/laplacian/garment_env.py`, `task: garment`)

Built on the gym's `build_jeans_scene` (livinglab_hq_v2 + LPR1 + flex garment on rack A shelf 3, gym
9e3220f). Same action interface as the rigid env (EE-local deltas in the `lpr1/mobile/base` frame, DLS-IK,
position servos) at 15 Hz = 267 physics substeps at the gym's mandatory 4 kHz. Reset writes the full qpos:
ready keyframe, mobile-base joints from the gym's A3 randomization ranges (no mocap in v2), garment
vertices jittered per env (port of `jitter_stack`), 1 s settle in 267-step chunks (a 4000-step CUDA graph
takes minutes to capture). Observations: native cameras of the W40 lpr1 datasets (wrists fisheye 320x240,
head pinhole 640x400). Reward: reach -> one pad -> both pads (+lift shaping) -> lifted (centroid >= 7 cm)
-> success after 3 held steps; drop off the shelf terminates. Prompt = the dataset string per arm.
Two fixes were needed on the gym side of things: the v2 IBL probe mixes float64/float32 (cast patch), and
the flex garment collides with all 23k scan boxes + 56 robot meshes by default (`cull_radius` gives the
garment its own collision bit shared only with the grasping hand and nearby scan geoms).

Throughput (2 envs, GPU idle, `scripts/garment_env_smoke.py`): env build 80-210 s, reset+settle 90-145 s,
then per control step physics 5.0-8.0 s + render 1.3-1.8 s. Physics cost is per batch, not per env
(4 envs: 5.1 s), and neither collision culling (8.0 -> 5.6 s) nor solver iterations 30 -> 8 change it
much, so it is the flex/nv=366 kernels themselves. For RL that means ~25 min per 128-step epoch with 16
envs per worker; the overnight garment PPO is therefore a 12-epoch pipeline demonstration, not a
converged run.

## 4. Company IL checkpoint for the garment task, evaluated in the new env

Choice: **UMICore-v0.17.9/checkpoint-30000** (converged, first run whose mix contains the W40 lpr1
"garment on the shelf" teleop sets, ~4.6 % weight; v0.17.11 has more weight but is mid-run). Control:
v0.16.10@50000, which has zero garment-shelf data. Prompt, verbatim from the dataset: "Pick up the
garment on the shelf. the left arm grasps and the right arm does nothing". Protocol: single package,
rack A shelf 3, gym A3 placement ranges, 150 steps (10 s), eval mode, one episode per GPU.

| checkpoint | seeds | success | closest TCP-garment (m) | pad contacts | gripper closes |
|---|---|---|---|---|---|
| UMICore-v0.17.9@30000 | 7 | 0/7 | 0.032-0.055 | 0 | max cmd 0.10 |
| UMICore-v0.16.10@50000 (no garment-shelf data) | 3 | 0/3 | 0.036-0.048 | 0 | max cmd 0.50 |

Both policies drive the left arm to the package's aisle-side edge in a side-pick posture and then hold
3-10 cm short of it with the gripper open for the rest of the episode; no pad ever touches the garment.
v0.17.9 seed 5 crashed in the pose-to-rotvec conversion on a NaN physics state (fixed: NaN counts as a
failed episode). Videos: `episode_v0179_garment_left_garment_seed<s>.mp4`,
`episode_v01610_garment_left_garment_seed<s>.mp4`. Placement variants for v0.17.9 seed 0: quarter-turn package (`yaw_offset_deg: 90`) -> same
hover, closest 4.0 cm, no contact (`episode_v0179_garment_left_yaw90_garment_seed0.mp4`); front
overhang 8 cm is rejected by the in-shelf jitter check (the gym's own jitter assumes the package inside
the board), not run.

Reading: the sim scene is not yet close enough to the W40 teleop scenes for the policy to commit to the
insertion (a 4 kHz flex package with 2 mm rubber pads whose side-pry presets the gym authors say all
failed after the pads were added). The RL run below starts from this hovering behaviour.

## 2. RL fine-tune of v0.16.10 on the new physics (v8 / v8b)

v8 = v5 optimisation settings (flow-SDE noise 0.6, lr 2e-5, 2 update epochs, critic warm-up 20) from
v0.16.10@50000 on gym 9e3220f with the v6 reward + orientation term and `boost` gripper mapping. First
launch (8 GPUs, 16 epochs) ran before the cond_add fix and was discarded; v8b = same on GPUs 0-3 with 16
envs (GPUs 4-7 serve the garment task), 100 epochs, ~3.3 min/epoch, log `lgym-ppo-staged-v8b.log`.
Train-mode success per epoch: 0-19 %, mostly 0-6 %, no trend through epoch 58 (rubber pads make the
grasp harder than on the old gym).

Balanced eval (8 seeds x 3 objects, boost, eval mode):

| policy | ROB-001 | ROB-003 | RTC-001 | all | lifts |
|---|---|---|---|---|---|
| v0.16.10@50000 (base) | 0/8 | 0/8 | 1/8 | 1/24 | 1 |
| v5@110 (old best, trained on the old gym) | 0/8 | 1/8 | 0/8 | 1/24 | 7 |
| v8b@50 | 0/8 | 0/8 | 2/8 | 2/24 | 5 |
| **v8b@100** | 0/8 | **3/8** | 2/8 | **5/24** | 4 |

Train-mode success averaged per 20 epochs: 2.6 %, 6.6 %, 5.5 %, 5.7 %, 3.9 %. The eval gain at step 100
(5/24 vs 1/24 for the base) is small but it is the first RL checkpoint that succeeds on a box (ROB-003,
the flat one) on the new physics. Success videos: `episode_v8b_100_gym9e3_boost_ROB-003_seed{4,6,7}.mp4`,
`episode_v8b_100_gym9e3_boost_RTC-001_seed{0,5}.mp4`; the base's only success is
`episode_base_gym9e3_boost_RTC-001_seed1.mp4`. Exported: `/home/thanh/models/umi-rl-ppo-v8b-step100`.

## 5. RL fine-tune of v0.17.9 on the garment env

Config `configs/rl/laplacian_ppo_gr00t_umi_garment_v1.yaml` (+ `configs/env/laplacian_garment_left_v1.yaml`):
v5 optimisation settings with flow-SDE noise 0.4, 8 GPUs x 16 envs, 128-step episodes, 12 epochs,
checkpoint every 3. Getting the batched flex scene through RLinf took five launches overnight, each a
real fix now in the repo: the gym's mujoco-warp CCD buffer (`naccdmax`) overflowed as soon as the hand
meshes touched the garment (contacts dropped, NaN state, SVD crash in the fork's rot6d decode); the
canonical 4 Nm gripper blows the flex up (now 2 Nm force limit + closure cap 0.92, the gym's own pick
settings); NaN observations must not reach the processor (NaN-safe decode + NaN counts as a failed
episode); and the RL config overrode the env's episode length (240 not divisible by the PPO batch).
The session that drove the runs was restarted at ~19:00 UTC, so the first clean launch is 2026-10-01
02:50 UTC; results are appended below as epochs land (~25 min each).

### Garment physics NaN: root cause and fix (2026-10-01)

Every batched garment run went NaN within the first episode. The sequence of findings:

1. NaN appeared on whole 16-world batches at once, long before the hand reached the garment, and
   persisted after reset. A 2-world poison test showed that a NaN world does not leak into its
   neighbour and that the masked reset does clear it, so neither leakage nor reset was the cause.
2. The persistence was the trainer: NaN log-probs from a NaN world's observation made the first PPO
   update's value loss and gradient norm NaN, after which every action was NaN. Guards now zero
   non-finite rewards/terms/states in the env and non-finite log-probs/values/chains in both the
   rollout output and the actor forward (commit c0 of this morning).
3. A state dump at the first NaN showed a benign world (max joint speed 0.4 rad/s, finite controls,
   0.02 rad command changes) whose entire robot tree was NaN one 267-substep frame later, starting with
   the mobile base and lift joints. Replaying that frame: the world alone is finite; the full 16-world
   batch is finite; a particular 8-world subset is NaN in every world; 8 kHz physics is NaN in all 16;
   100 Newton iterations move the NaN to a different world. The only overflow the run ever reported was
   "solver iterations limit reached". Conclusion: mujoco-warp's Newton solver fails to converge on
   this robot+flex scene and fails to NaN, with the outcome depending on contact ordering in the batch.
4. On the reproducing subset: Newton 30 and 200 iterations -> NaN; CG 100 and CG 300 -> finite for 6
   frames. `GarmentRLConfig` now defaults to `solver: CG, solver_iterations: 100`.

Along the way the garment env also gained: rubber-pad-only flex collision (the thousand-face finger
meshes made the flex CCD overflow), per-world CCD buffers, 2 Nm grip force / 0.92 closure cap (the
gym's own pick settings), per-step EE delta clamps (2 cm / 0.1 rad), and in-place reset of NaN worlds.

### Garment PPO v1 results (CG solver run, started 2026-10-01 06:21 UTC)

128 envs (8 x 16), 128-step episodes, ~25 min per epoch, checkpoints every 3. Training is finite
throughout (one NaN world in the first 384 episodes, reset in place). Per epoch: KL 0.088 -> 0.035,
critic explained variance -11.6 -> 0.36, stage_max 2.36 -> 2.55, grasp 0.

| checkpoint | success (8 seeds, 150 steps) | closest TCP-garment | one pad touching | both pads | return |
|---|---|---|---|---|---|
| v0.17.9@30000 zero-shot | 0/7 | 3.2-5.5 cm | 0/7 | 0/7 | 165-189 |
| PPO step 3 | 0/8 | 3.4-5.1 cm | **8/8** | 0/8 | 262-327 |

After three epochs the policy consistently brings a pad onto the garment (the first shaped stage) but
does not close or lift. Videos: `episode_garment_ppo_step3_garment_seed<s>.mp4`. Steps 6, 9, 12 are
exported and evaluated the same way as they land (`garment_ckpt_pipeline.sh`).

