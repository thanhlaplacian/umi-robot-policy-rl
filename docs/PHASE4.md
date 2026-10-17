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

## Results

### Rigid pick-and-place (strict placement success, 3 objects x 8 seeds)

| checkpoint | ROB-001 | ROB-003 | RTC-001 | all |
|---|---|---|---|---|
| v0.20.3@8000 baseline | 2/8 | 3/8 | 2/8 | **7/24** |

For reference, v0.16.10@50000 and its RL fine-tune v8b@100 scored 3/24 each on this protocol (older
gym pin). Baseline videos: `episode_v0203_base_rigid_<object>_seed<s>.mp4`.

