# Phase 4 (2026-10-17): RL fine-tune of UMICore v0.20.3 on both sim tasks

Request: fine-tune v0.20.3 with RL on the rigid pick-and-place env and the garment env, save every
model under `/home/thanh/models`, and report the improvement over the baseline. 12-hour budget.

## Setup

- **Base:** `/data/models/UMICore-v0.20.3/checkpoint-8000` (LPR1-only mix, 100 % `umi_bimanual_lpr1`
  datasets W36-W41 teleop + W38 sim; 3 cameras, 16-step cumulative EEF, loss 0.0118 at step 8000).
  New embodiment tag `umi_bimanual_lpr1` (id 8) -> fork bumped to a08b307 (+ SDK ecb9e3c); the
  checkpoint loads with zero missing/unexpected tensors. Model config `configs/model/gr00t_n1d7_umi_lpr1.yaml`.
- **Gym:** laplacian-gym ad3c08c (LPR1 v2 wrist, 6.6 cm longer TCP than every pre-10-17 run).
- **Rigid env v9** (`configs/env/laplacian_pick_place_staged_1obj_v9.yaml`): v8 reward (staged + align +
  orientation term, `boost` gripper) under the strict placement success (inside the basket volume,
  released, held 5 steps). Eval protocol: 3 objects x 8 seeds, eval mode, one process per GPU.
- **Garment env** (`configs/env/laplacian_garment_left_v1.yaml`): CG solver, pads-only flex collision,
  128-step episodes. Eval protocol: 8 seeds, 150 steps, strict garment success (both pads + 7 cm lift).
- **PPO:** `configs/rl/laplacian_ppo_gr00t_umi_v0203_rigid.yaml` (GPUs 0-3, 16 envs, noise 0.4, lr 1.5e-5,
  2 update epochs, 180 epochs max, checkpoint every 10) and `..._v0203_garment.yaml` (GPUs 4-7, 64 envs,
  noise 0.4, lr 2e-5, 24 epochs max, checkpoint every 3). Both with the cond_add fix and NaN guards.
- **Exports:** `/home/thanh/models/umi-rl-v0203-rigid-step<N>` and `/home/thanh/models/umi-rl-v0203-garment-step<N>`
  (fork-loadable HF dirs), produced by `v0203_ckpt_pipeline.sh` as checkpoints land; each is evaluated with
  the protocol above. Logs: `lgym-ppo-v0203-rigid.log`, `lgym-ppo-v0203-garment.log`,
  `v0203_ckpt_pipeline.log`. Videos: `/home/thanh/rlinf-runs/logs/umi-rl/gym_episode/episode_v0203_*`.

## Integration notes for gym ad3c08c

Both PPO runs crashed at their first reset with a CUDA illegal memory access inside the gym's shading
kernel (`rendering/lighting.py shade`). Reproduced outside RLinf: 2 and 3 envs render, 4 do not (the
LPR1 v2 wrist adds 19 % visual and 2.7x collision faces). Workaround in `LaplacianRLEnv.render_all`:
render each camera in sub-batches of `render_chunk` (default 2) envs via the gym's `render_enabled`
mask; the garment env additionally slices the flex vertex tensor per chunk because the gym's skinning
reads the full batch on subset renders. Cost: rigid 4 envs 0.51 s/step (was 0.36 s unchunked at 3 envs),
garment 16 envs 9.7 s/step (render 5 s in 8 chunks, physics 4.5 s). Runs relaunched 14:20 (rigid) and
14:26 (garment) UTC.

## Results

### Rigid pick-and-place (strict placement success, 3 objects x 8 seeds)

| checkpoint | ROB-001 | ROB-003 | RTC-001 | all |
|---|---|---|---|---|
| v0.20.3@8000 baseline | 2/8 | 3/8 | 2/8 | **7/24** |
| PPO rigid step 10 (noise 0.4, lr 1.5e-5) | 0/8 | 1/8 | 2/8 | 3/24 |

For reference, v0.16.10@50000 and its RL fine-tune v8b@100 scored 3/24 each on this protocol (older
gym pin). Baseline videos: `episode_v0203_base_rigid_<object>_seed<s>.mp4`.

### Garment pick (strict garment success: both pads on the package and 7 cm lift; 8 seeds, 150 steps)

| checkpoint | success | one pad touching | both pads | closest TCP-garment | return |
|---|---|---|---|---|---|
| v0.20.3@8000 baseline | 0/8 | 8/8 | 0/8 | 2.6-3.7 cm | 212-320 |

v0.17.9@30000 (old base) never touched the package (0/7, 3-5 cm away, return 164-189); v0.20.3 reaches
the package edge with one pad every time but does not close. Videos: `episode_v0203_base_garment_garment_seed<s>.mp4`.
### Run log

- 15:27 UTC: the garment run (noise 0.4, lr 2e-5, 2 update epochs) had approx_kl 2.9 after its first
  update and 0.58 after the second, clip fraction 0.39, stage_max falling 2.9 -> 1.7: a policy collapse,
  not a fine-tune. Stopped and relaunched as `garment_b` (lr 5e-6, noise 0.3, 1 update epoch); exports go
  to `/home/thanh/models/umi-rl-v0203-garment-b-step<N>`. The rigid run's KL is 0.05-0.09 per update
  (clip fraction 0.16-0.25) and its train-mode strict success rose from 0 to 6-24 % by epoch 11, so it
  continues; `rigid_b` (same gentler settings) is prepared in case step 20 also sits below the baseline.
- 15:54 UTC: rigid-a (noise 0.4, lr 1.5e-5, 2 update epochs) stopped after saving step 20: its train-mode
  strict success peaked at 24 % around epoch 8-11 and fell back to 0-6 % by epoch 19 while step 10 scored
  3/24 in eval. `rigid_b` (lr 5e-6, noise 0.3, 1 update epoch) launched from the base on GPUs 0-3;
  exports go to `/home/thanh/models/umi-rl-v0203-rigid-b-step<N>`. Rigid-a step 20 is still evaluated.
