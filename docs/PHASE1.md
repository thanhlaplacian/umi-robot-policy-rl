# Phase 1: offline RL with STEAM advantages through RLinf

Status (2026-09-28): increment 1 **done** (8-GPU RLinf SFT smoke trained, checkpointed, exported
and reloaded by the fork); increment 2 wired (`advantage_weight_suboptimal`), not yet run.

## What exists, measured

- **STEAM labels** (`scripts/steam_label_audit.py`, mix v0.16.9, label version v0.2.16):
  26 of 77 datasets carry `meta/steam_advantage_v0.2.16.npz` (TELE W26-W31, TELE2-rebot W33-W34,
  UMI W25/W30, UMI2 W32, Cosmos-DAgger W29/W31). That is 4.78 M of 43.98 M frames (10.9 %) and
  28.4 % of the mix weight. Newer UMI2 (W33+) and rebot (W35+) sets and the SIM2/lpr1 sets have no
  labels. Among labelled frames 92.3 % have A >= 0 (the `steam_optimality` cut used in the fork);
  the rebot sets are the exception (68-82 % optimal, median A 0.30-0.50) while UMI/DAgger sets
  sit at 88-97 %.
- **What the fork already does with the labels** (per-frame `optimality` flag, threshold per
  dataset): CFGRL conditioning (`use_optimality_conditioning`) and AWR-style loss weighting
  (`advantage_weight_suboptimal`, binary, weighted mean). Both are off in the v0.16.9 recipe.
- **What RLinf offers**: its IQL path is D4RL/MLP only; its STEAM stage 3 (CFG) is pi0.5 only;
  its FSDP SFT stack (`FSDPVlaSftWorker` + `SFTRunner`) is model-agnostic and takes any loader
  whose batches the model's `forward(forward_type=SFT, data=...)` understands.

## Design decision

Build the offline path on RLinf's SFT stack with the fork's data pipeline, in three increments:

1. **Plain SFT parity** (this phase's smoke): the company model trains under RLinf FSDP2 on 8 GPUs
   from LeRobot data with the STEAM flag stamped, loss identical in form to the fork's trainer.
   Value: proves the distributed loop, checkpointing and logging; gives a baseline curve.
2. **Advantage-weighted flow matching (AWR / filtered BC)**: per-sample weight `w = 1` for
   optimal frames and `w = advantage_weight_suboptimal` otherwise (the fork's own rule), or a
   continuous `exp(A / beta)` once labels justify it. Weights ride through the batch dict, no
   worker change. This is offline RL in the RWR/AWR family and needs no value function.
3. **Value-based offline RL (IQL / RECAP-style)**: a state-value head on the frozen VLM features
   (RLinf already has a value head on the RL model, `add_value_head`) trained on STEAM-derived
   returns, then advantage-weighted policy extraction. Only worth it after (2) shows signal.

Why not RLinf's CFG path: it conditions pi0.5 by appending text to the prompt; the company model
already has a learned optimality embedding for the same purpose, so CFGRL is better run in the
fork's trainer. Why not RLinf's IQL path: it has no image/chunk support.

## Done in this phase

- `umi_rl/offline/data.py`: `build_processor` (train mode), `build_dataset` (SDK
  `DatasetFactory` -> `ShardedMixtureDataset`, STEAM stamping per dataset), `build_loader`.
  One batch from `26-W33-TELE2-rebot-batch1` carries `optimality` `[0,1,0,1,1,1,1,1]`, `action`
  `(B,16,132)`, `action_mask`, `state`, VLM tokens/pixels, progress targets.
- `UmiGr00tN1d7ForRL.sft_forward` / `forward(forward_type=SFT)`: the fork's training forward on
  RLinf's model object (RLinf's RL head overrides `prepare_input`/`forward`, so the fork's
  versions are called explicitly). Loss 0.0147 on the first batch, backward ok, 9.2 GiB at B=8
  in bf16.
- `umi_rl/offline/sft_worker.py` + `scripts/train_offline_sft.py` + `scripts/run_offline_sft.sh`
  + `configs/offline/umi_sft_smoke.yaml`: RLinf FSDP2 SFT on 8 GPUs (`actor: 0-7`, micro 4,
  global 64, fp32 master weights under bf16 autocast, gradient checkpointing, DCP checkpoints
  plus full weights).

## Smoke result (`configs/offline/umi_sft_smoke.yaml`, run `umi-rl-offline-sft-smoke`)

| Item | Value |
|---|---|
| Setup | 8 x GPU, FSDP2 full-shard, fp32 master + bf16 autocast, gradient checkpointing, micro 4 x accum 2 = 64 samples/step |
| Steps | 20 (dataset `26-W33-TELE2-rebot-batch1`, STEAM v0.2.16 stamped, lr 1e-5) |
| train/loss | 0.0139 -> 0.0101 (range 0.009-0.018, noisy at this batch size) |
| train/progress_loss | 0.030 -> 0.016 |
| train/grad_norm | 0.02-0.035 |
| time/step | 0.8-1.0 s (first step 3.4 s; last step 32.7 s includes the checkpoint) |
| GPU memory | ~47 GB per GPU during training |
| Checkpoint | `.../checkpoints/global_step_20/actor/` = `dcp_checkpoint` 26 GB + `model_state_dict/full_weights.pt` 13 GB + `data.pt` + `rng.pt` |
| Export | `scripts/export_hf_checkpoint.py` -> `/home/thanh/models/umi-rl-offline-sft-smoke-step20` (2 safetensors shards, 6.4 GiB bf16); `AutoModel.from_pretrained` loads it, tensor names identical to the base (0 missing / 0 unexpected), action-head weights moved by ~2e-4, backbone unchanged (frozen in the recipe) |
| Logs | TensorBoard `/home/thanh/rlinf-runs/logs/umi-rl/tensorboard`, W&B project `finetune-gr00t-n1d7` |

How to run: `ray start --head --port=6379 --dashboard-host=127.0.0.1 --dashboard-port=8265` once
(the venv's Ray), then `bash scripts/run_offline_sft.sh umi_sft_smoke`; stop a run with
`scripts/kill_offline_sft.sh`.

## Next

1. Score the exported step-20 checkpoint and the base with the fork's open-loop holdout eval
   (TELE2-rebot VAL2) to confirm the RLinf loop does not regress the policy.
2. Increment 2: `data.advantage_weight_suboptimal` -> per-sample weights in `sft_forward`
   (`loss = sum(w * l) / sum(w)`), logged `advantage_weight_sum`; run A/B (weight 1.0 vs 0.3 vs
   0.0) on the labelled mix for 5k steps each, compare holdout ADE and rollout on the rebot.
4. Label coverage: the newest rebot/UMI2 sets are unlabelled; scoring them needs the fork's
   STEAM `score.py` with `/data/models/steam/v0.2.16` (writes into the shared dataset tree, so
   agree with the team first).
