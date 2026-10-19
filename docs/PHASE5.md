# Phase 5: RL fine-tuning of the UMICore WAM v0.18 (model_type `umi_wam`)

Plan: [PLAN_WAM.md](PLAN_WAM.md). Model `/data/models/UMICore-v0.18.39/checkpoint-3000`, model config
`configs/model/umi_wam_v018.yaml`, policy `umi_rl/wam/policy.py` (`UmiWamForRL`), run config
`configs/rl/laplacian_ppo_wam018_rigid.yaml`. RLinf fork `integration` 6094305f adds `SupportedModel.UMI_WAM`.

## W0. Load, parity, baseline

- `scripts/wam_parity_check.py`: the umi_wam eval path against the fork's `Gr00tPolicy` on the same model, two
  simulator batches of 4 envs, same seed: **max |diff| 0.0** on the decoded 14-dim action (noise-to-noise spread
  1.7e-2 / 2.7e-2).
- Baseline, strict rigid protocol (3 objects x 8 seeds per draw, boost gripper): see the table in W3.

## W1. RL surface on one GPU

| check | result |
|---|---|
| `wam_cond_parity.py`: train sampler at noise 0 vs `get_action` (same VAE chunking) | max diff 0.0 |
| `wam_rl_path_smoke.py`: rollout then replay, micro-batch 8 | ratio exactly 1, values identical, grads finite, 11.3 GiB peak (with or without checkpointing), rollout 0.16-0.24 s, replay + backward 0.2-0.35 s |
| `wam_sft_smoke.py`: `sft_forward` on TELE W36 batch1 (live und tower for dataset prompts) | finite loss and grads, action loss 6e-4, 10.3 GiB at batch 2 |

## W2. PPO smoke through RLinf (2 epochs, GPUs 4-7)

The first smoke ran but showed approx_kl 0.093 and clip fraction 0.24 **with lr 0** (critic warmup): the
FSDP actor's replay disagreed with the HF rollout before any update. Three causes, each measured with
probes and fixed in the policy:

| cause | size | fix |
|---|---|---|
| FSDP `buffer_dtype: bf16` casts the RoPE `inv_freq`; WAM position ids reach ~1.5e4 | per-element log-ratio 4e-3 to 2e-2 (grows with the denoising step) | `inv_freq` pinned to fp32 in the RL paths |
| batched VAE encodes round each frame by its batch position (2e-2 on the latents) | chunk-level log-ratio up to 0.22 on a shuffled replay | one frame per VAE call in the RL paths; eval keeps the checkpoint's `wam_vae_chunk` (deployment numerics) |
| FSDP mixed precision casts every float forward input to bf16 (stored chains, `prev_logprobs`) | ratio_abs 0.13 after the two fixes above | chain states rounded to bf16 before they are scored; `prev_logprobs` returned for the noisy step only and kept by the actor (UMI_WAM is not in the actor's prev_logprobs lists) |

The replay is exact only at the rollout batch size (GEMM rounding differs with the batch size): actor
`micro_batch_size` = env batch per rollout worker = 8; the policy warns on a mismatch.

| smoke | lr | approx_kl | clip fraction | ratio_abs |
|---|---|---|---|---|
| before the fixes | 0 | 0.093 / 0.072 | 0.235 | 0.285 |
| fp32 RoPE only | 0 | 0.022 / -0.002 | 0.10 | 0.14 |
| + per-frame VAE | 0 | 0.007 / -0.009 | 0.10 | 0.125 |
| + bf16-exact chains, actor-side prev_logprobs | 0 | -1.6e-7 / 8.8e-8 | 0.0 | 4.4e-6 |
| same, critic warmup off | 1e-5 | 0.011 / 0.008 | 0.079 / 0.054 | 0.10 / 0.09 |

Export: `scripts/export_hf_checkpoint.py` writes the base's 400 tensors (value head dropped, bf16) plus the
base's non-weight files (`text_tokenizer/`, und index and reference); the fork's `Gr00tPolicy` loads it.

The N1.7 runs (`gr00t_n1d7_umi`) pass chains and `prev_logprobs` through the same FSDP forward; their
approx_kl of ~0.018 at lr 0 (PHASE4 RLinf bump) is probably the same bf16 cast. Not changed here.

## W3. The real run

`laplacian_ppo_wam018_rigid`, launched 2026-10-19 14:49 UTC on GPUs 4-7, 120 epochs, checkpoint every 10.
Exports `/home/thanh/models/umi-rl-wam018-rigid-step<N>`, evaluated by `~/rlinf-runs/tools/wam018_ckpt_pipeline.sh`
(24 strict episodes per checkpoint on GPUs 0-3, PHASE4 draw rules).

Results: pending.
