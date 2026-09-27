# Phase 0 report: RLinf loads and reproduces the company GR00T N1.7 policy

Date: 2026-09-28. Checkpoint under test: `/data/models/UMICore-v0.16.9/checkpoint-3000`
(embodiment tag `umi_bimanual_v2`, 16-step chunk, 20-dim normalized action, 14-dim state,
two wrist views + optional `cam_head`). Fork pinned at `a379960`, RLinf at `ee2cab65`.

## Result

| Check | Script | Outcome |
|---|---|---|
| Environment | `scripts/setup_env.sh` | venv `/home/thanh/venvs/umi-rl`: torch 2.11.0+cu130, transformers 4.57.3, flash-attn 2.8.3, `gr00t` imported from the fork, `umi_data_sdk`, `rlinf` editable |
| Model load through RLinf | `umi_rl/model.py` (`model_type: gr00t_n1d7_umi`) | loads in ~40 s, 3.15 B params, `valid_action_dim 20`, 3 views, processor tags `umi_bimanual_v2 / umi_bimanual / oxe_droid_eef` |
| Action parity vs the fork's `Gr00tPolicy` | `scripts/parity_check.py` | **max abs diff 0.0** on all six action keys, TELE2-rebot ep 0 (steps 0/40/80) and UMI2 ep 3 (steps 0/50/100), batch of 3, same seed. Seed-to-seed spread for scale: 7e-3 to 2e-2 |
| Input parity (tokens, pixels, state) | `scripts/parity_inputs.py` | `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, `state` identical between `Gr00tPolicy` (`processor(messages)`, the deployment path) and RLinf (`process_observation`) |
| RL surface (train-mode rollout, actor forward/backward) | `scripts/rl_path_smoke.py` | chains `(B, 5, 16, 132)`, log-probs `(B, 4, 16, 20)`, values finite; replayed log-prob ratio mean 1.0009; backward reaches 545 parameter tensors; 9.3 GiB peak for B=4 on one GPU |
| Prompt padding sensitivity | `scripts/padding_invariance.py` | 16 right-pad tokens move the decoded action by at most one bf16 ulp (3.9e-3 on gripper, 7e-4 m on position), same as plain batch-size numerics, with or without the VLSA mask patch |

## What had to change, and where

Nothing in the fork. Two things in RLinf's rollout wrapper were reproduced *differently* from the
fork's deployment path, both overridden in `umi_rl.model.UmiGr00tN1d7ForRL` (RLinf itself is
untouched):

1. **Raw state rounded to bf16 before normalization** (`_prepare_rollout_observation`). RLinf
   does this to mimic its LIBERO training data. The fork normalizes the float32 state and only
   then casts to bf16, in training and serving alike. Effect: one bf16 ulp on the normalized
   state, ~1e-3 on decoded actions.
2. **`torch.autocast(bf16)` around eval sampling** (`_get_action_from_normalized_input`). The
   fork's `Gr00tPolicy` and planner-vla run the bf16 model under `inference_mode` only. Effect:
   ~1e-3 on decoded actions.

Plus two loader shims:

- `umi_rl/compat.py` aliases `gr00t.data.embodiment_tags` to `umi_data_sdk.core.embodiment_tags`
  (the fork has no `gr00t.data`; RLinf imports the enum from there).
- The processor is built by the fork's own `Gr00tN1d7Processor.from_pretrained`, not by RLinf's
  `Gr00tN1d7Processor(**processor_kwargs)`: the checkpoint was written by a newer fork commit and
  carries the legacy key `optional_views_last`, which only the fork's loader knows to alias.

Registered without editing RLinf: `model_type gr00t_n1d7_umi` (`register_model`), and
`obs_converter_type umi_bimanual` (entries in `OBS_CONVERSION` / `ACTION_CONVERSION_N1D7`).

## Conventions the RL side must keep

- Observation batch (`env_obs`): `states` float `(B, 14)` in dataset order
  `[grip_R, pos_R(3), rotvec_R(3), grip_L, pos_L(3), rotvec_L(3)]`; images uint8 `(B, H, W, 3)`
  under `wrist_left_images` / `wrist_right_images` / optional `head_images` (or RLinf's real-env
  `main_images` + `extra_view_images`); `task_descriptions` list of str.
- Decoded action `(B, 16, 14)`: per-step EE-local forward deltas (xyz + axis-angle) with absolute
  grippers, the same convention as the parquet `action` column and planner-vla's input.
- The normalized RL action space (log-probs, PPO ratio) is the model's `(16, 20)` slice:
  gripper, xyz, rot6d per arm, chunk-start-relative, min-max in [-1, 1].
- `padding_value: 0`; with the measured padding insensitivity, mixed prompts in a batch are
  acceptable for this checkpoint. The VLSA mask patch stays optional (`GR00T_VLSA_APPLY_MASK=1`).
- `GR00T_TEXT_PADDING_SIDE` unset (left) matches the checkpoint's training.
- Video decoding in this venv uses `pyav` (`torchcodec` is not installed; the fork defaults to it
  inside its own container).

## Not covered by phase 0

- No environment exists yet: rollouts came from dataset frames. Real-robot or simulator envs
  are phase 2.
- The fork's `advantage_weight_suboptimal` / CFGRL conditioning are inactive on this checkpoint
  (`use_optimality_conditioning: false`); phase 1 decides how STEAM labels reach the RLinf loss.
- The `train`-mode rollout still runs under RLinf's autocast (matches the fork's bf16 training);
  only the eval path was aligned with deployment.
