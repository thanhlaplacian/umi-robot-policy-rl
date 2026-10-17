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
| PPO rigid-a step 10 (noise 0.4, lr 1.5e-5) | 0/8 | 1/8 | 2/8 | 3/24 |
| PPO rigid-a step 20 | 1/8 | 0/8 | 3/8 | 4/24 |
| PPO rigid_b step 10 (noise 0.3, lr 5e-6, 1 update epoch; 3 actor updates past warm-up) | 1/8 | 0/8 | 5/8 | 6/24 |
| PPO rigid_b step 20 | 1/8 | 1/8 | 4/8 | 6/24 |
| PPO rigid_b step 30 | 3/8 | 1/8 | 3/8 | 7/24 |
| PPO rigid_b step 40 | 2/8 | 1/8 | 3/8 | 6/24 |
| PPO rigid_b step 50 | 0/8 | 2/8 | 4/8 | 6/24 |
| PPO rigid_b step 60 | 3/8 | 1/8 | 2/8 | 6/24 |
| PPO rigid_b step 70 | 1/8 | 1/8 | 3/8 | 5/24 |
| PPO rigid_b step 80 (final) | 0/8 | 2/8 | 5/8 | 7/24 |
| PPO rigid_c step 10 (32 envs, lr 1e-5, noise 0.3) | 1/8 | 1/8 | 3/8 | 5/24 |
| PPO rigid_c step 20 | 2/8 | 2/8 | 7/8 | **11/24** |

Gripper-mapping diagnostic: the same baseline with the plain `linear` mapping instead of `boost` scores
5/24 (ROB-001 0, ROB-003 1, RTC-001 4), so the protocol's `boost` mapping is not handicapping v0.20.3.

48-episode comparison (seeds 0-7 plus a second draw on seeds 8-15):

| checkpoint | seeds 0-7 | seeds 8-15 | total |
|---|---|---|---|
| v0.20.3@8000 baseline | 7/24 | 7/24 | **14/48 (29 %)** |
| PPO rigid_b step 20 | 6/24 | 4/24 | 10/48 (21 %) |
| PPO rigid_c step 20 | 11/24 | 6/24 | **17/48 (35 %)** |

96-episode comparison (draws on seeds 0-7, 8-15, 16-31; 32 episodes per object):

| checkpoint | ROB-001 | ROB-003 | RTC-001 | total |
|---|---|---|---|---|
| v0.20.3@8000 baseline | 2/32 | 6/32 | 14/32 | 22/96 (23 %) |
| PPO rigid_c step 20 | 3/32 | 3/32 | **22/32** | **28/96 (29 %)** |

The gain is on the cylinder (RTC-001, 44 % -> 69 %) with the boxes unchanged within noise; the overall
+6 points is about one standard error for 96 episodes (the per-draw results were 11 vs 7, 6 vs 7, 11 vs 8).

For reference, v0.16.10@50000 and its RL fine-tune v8b@100 scored 3/24 each on this protocol (older
gym pin). Baseline videos: `episode_v0203_base_rigid_<object>_seed<s>.mp4`.

### Garment pick (strict garment success: both pads on the package and 7 cm lift; 8 seeds, 150 steps)

| checkpoint | success | one pad touching | both pads | closest TCP-garment | return |
|---|---|---|---|---|---|
| v0.20.3@8000 baseline | 0/8 | 8/8 | 0/8 | 2.6-3.7 cm | 212-320 |
| PPO garment_b step 3 (lr 5e-6, noise 0.3) | 0/8 | 8/8 | 0/8 | 3.1-3.9 cm | 186-311 |
| PPO garment_b step 6 (final) | 0/8 | 8/8 | 0/8 | 2.9-4.3 cm | 280-320 |

No garment checkpoint improves on the baseline: every policy reaches the package edge with one pad and
never closes, and no training batch ever contained a two-pad contact, so there was no success signal
to learn from. Videos: `episode_v0203_garment_b_step{3,6}_garment_seed<s>.mp4`.

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
- ~18:15 UTC: garment_b stopped after exporting step 6. Across every garment attempt (v0.17.9 x 12 epochs,
  v0.20.3 garment-a, garment_b) the training batches never contained a single two-pad contact, so the
  policy has no success signal to climb and 14 more epochs at 30 min each could not change that within
  the budget. GPUs 4-7 were reassigned to `rigid_c` (32 envs per update instead of 16, lr 1e-5, noise 0.3,
  1 update epoch) to give the rigid task, where the baseline already succeeds 29 %, a second variant.
  Garment exports: `/home/thanh/models/umi-rl-v0203-garment-b-step{3,6}`.
## How to evaluate the saved models yourself

Every checkpoint is a fork-loadable HF directory under `/home/thanh/models/umi-rl-v0203-<run>-step<N>`
(`rigid-b`, `rigid-c`, `rigid`, `garment-b`). Inside the container (`bash scripts/in_container.sh -g <gpu> ...`):

```bash
# rigid pick-and-place, balanced strict protocol (24 episodes, one process per GPU)
python scripts/eval_queue.py --ckpt /home/thanh/models/umi-rl-v0203-rigid-b-step30 --tag mytag --gpus 0-7 \
    --extra "--model-cfg configs/model/gr00t_n1d7_umi_lpr1.yaml"
python scripts/eval_queue.py --summary v0203_base_rigid,mytag
# garment, 8 seeds
python scripts/eval_queue.py --task garment --ckpt /home/thanh/models/umi-rl-v0203-garment-b-step6 --tag mytag_g \
    --seeds 0-7 --gpus 0-7 --env-cfg '{"cull_radius":0.5}' --extra "--max-steps 150 --model-cfg configs/model/gr00t_n1d7_umi_lpr1.yaml"
# one episode with video
python scripts/gym_episode_video.py --seed 0 --tag demo --ckpt <dir> --model-cfg configs/model/gr00t_n1d7_umi_lpr1.yaml \
    --env-cfg '{"spawn_count":1,"gripper_mode":"boost","gripper_thresh":0.3,"object_types":["ROB-003"]}'
```

The base is `/data/models/UMICore-v0.20.3/checkpoint-8000` with the same commands.

## Conclusion

(written when the runs end)
- 21:29 UTC: rigid_b stopped after exporting step 80 (eight checkpoints at 5-7/24, never above the
  baseline). GPUs 0-3 now serve evaluation only: a third and fourth draw (seeds 16-31) for the baseline and
  for rigid_c step 20, giving 96 episodes each. rigid_c keeps training on GPUs 4-7 with checkpoints every 10.
