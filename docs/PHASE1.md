# Phase 1: offline RL with STEAM advantages through RLinf

Status: **in progress** (2026-09-28). The data path and the 8-GPU training loop are wired; see
"Done" and "Next" below.

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

## Next

1. Finish the smoke (20 steps), read the loss curve and the checkpoint layout.
2. Converter: DCP/full-weights checkpoint -> HF-style checkpoint dir (safetensors + the copied
   processor files) so the fork's `open_loop_eval_eef.py` can score it on the TELE2 holdout.
3. Increment 2: `data.advantage_weight_suboptimal` -> per-sample weights in `sft_forward`
   (`loss = sum(w * l) / sum(w)`), logged `advantage_weight_sum`; run A/B (weight 1.0 vs 0.3 vs
   0.0) on the labelled mix for 5k steps each, compare holdout ADE and rollout on the rebot.
4. Label coverage: the newest rebot/UMI2 sets are unlabelled; scoring them needs the fork's
   STEAM `score.py` with `/data/models/steam/v0.2.16` (writes into the shared dataset tree, so
   agree with the team first).
