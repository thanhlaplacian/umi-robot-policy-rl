# Plan: RL fine-tuning of the UMICore WAM (v0.18, `backbone_version: wam`) with RLinf

Context (verified 2026-10-19 against fork a08b307 and RLinf a78b6d7):

- `UmiWam` (`src/gr00t/model/umi_wam/`) is a Cosmos3-Edge generation tower (28 layers, hidden 2048,
  1.43 B trainable) plus a frozen Wan VAE encoder. There is no VLM: the prompt enters as a cached
  instruction K/V row (`und_cache.py`), looked up by the sha1 of the token ids of a JSON caption.
  Latest checkpoint `/data/models/UMICore-v0.18.39/checkpoint-3000`: embodiment `umi_bimanual_lpr1`,
  3 cameras at 224 x 320, 16-step chunk, `wam_infer_steps 4`, `wam_infer_shift 1.0`,
  `wam_attention two_pass_block` (= single-pass serving), `wam_action_tokens joint` (one 64-dim token
  per step, 20 real dims: R in 0-9, L in 10-19), progress + arm-active heads whose predictions are
  added to the action-token embedding (`_serve_heads` -> `cond`).
- Serving (`get_action`): `_observation_kv` once per observation (und prefix + O tokens, per-layer
  K/V), then per Euler step `_serve_action_embed -> trunk.run -> unembed_actions`, with
  `x <- x + (sigma_next - sigma) * v`, sigmas descending 1 -> 0, `v = eps - a`. Packed output via
  `_pack_hands` -> `[B, 16, 20]`; `UmiWamProcessor.decode_action` turns that into the 14-dim deltas.
- Feasibility (2026-10-18, `~/rlinf-runs/tools/wam_infer_check.py`): loads in the umi-rl venv (3.0 GiB
  bf16); `Gr00tPolicy.get_action` B=1 168 ms, B=32 1.1 s (peak 5.4 GiB); a gradient pass through the
  4-step Euler loop at micro-batch 8 takes 4.2 s, peak 14.1 GiB, finite grads. The obs converter,
  processor and decode already work with `umi_rl.converters`.
- RLinf a78b6d7 ships `cosmos3` and `fastwam` policies, but both implement `ForwardType.SFT` and
  eval only (no flow-SDE log-probs, no `default_forward`). The N1.7 RL head
  (`gr00t_n1d7/gr00t_action_model.py`: `sample_mean_var_val`, `get_rl_action`, replay `forward`,
  `ValueHead`) is the reference; it cannot be reused directly because it drives
  `action_encoder` / DiT / `action_decoder` submodules the WAM does not have.
- Blocker outside the code: the shared und caches `/data/models/wam_und_cache/*/und_kv.safetensors`
  are mode 600 owned by ubuntu. Workaround in place: a private cache for the three sim prompts at
  `/home/thanh/models/wam_und_cache/v0203-sim-prompts`; ask ubuntu for `chmod 644`.

Flow convention. RLinf parameterises `x_t = (1 - t) x0 + t x1` with `x0` noise, `x1` clean and
`v = x1 - x0`; the WAM parameterises `x = (1 - sigma) a + sigma eps` with `v = eps - a`. So
`t = 1 - sigma` and `v_rlinf = -v_wam`; with `shift 1.0` the WAM schedule is exactly RLinf's
`linspace(0, 1, 5)`. The flow-SDE step of `sample_mean_var_val` (mean weights and
`std = sqrt(delta) * noise_level * sqrt((1 - t) / t)`) therefore carries over per step, with `t` and
`delta` read from the checkpoint's sigma schedule so a shifted schedule would still be correct.

## What gets built

| Piece | Where | Notes |
|---|---|---|
| `UmiWamForRL(UmiWam, BasePolicy)`, `model_type: umi_wam` | `umi_rl/wam/policy.py`, `umi_rl/wam/get_model.py` | registered with `register_model`; imported from `umi_rl.model` so the `.pth` hook covers Ray workers |
| rollout: `predict_action_batch(env_obs, mode)` | same | obs convert (`umi_bimanual`) -> `process_observation` -> collate -> `_observation_kv` (no grad) -> `_serve_heads` -> 4-step loop with RLinf's flow-SDE mean/std in the sigma convention, noise injected on one sampled denoise index (train) or none (eval, which calls the fork's `get_action` unchanged for bit parity). Chains stored packed `[B, 5, 16, 20]` via `_pack_hands`, log-probs `[B, 4, 16, 20]`, `forward_inputs` = chains, denoise_inds, the processor tensors (`wam_wrist`, `wam_view3*`, `wam_state`, `und_row`, `embodiment_id`) |
| actor replay: `default_forward(forward_inputs, prev_logprobs, ...)` | same | re-run `_observation_kv` with grad (config switch `detach_obs_kv` to train only the action path), unpack `chains[idx]` with `_hand_tokens`, recompute mean/std and the log-prob of `chains[idx+1]`, value from the pooled O hidden + state tokens; returns `logprobs`, `prev_logprobs`, `values`, `entropy=None` like the N1.7 wrapper |
| value head | same | RLinf `modules.value_head.ValueHead(input_dim = 2048 + n_items * 64)`, fed by the mean over valid O tokens of `h_o` concatenated with `wam_state`; fresh init, `critic_warmup_steps` as in rigid_c |
| NaN guards and `sft_forward` | same | copy the guards of `UmiGr00tN1d7ForRL`; `sft_forward` = `UmiWam.forward(inputs)["loss"]` (offline path, later phase) |
| RLinf fork, one commit on `integration` | `rlinf/config.py`, actor worker, HF rollout worker | `SupportedModel.UMI_WAM` joins the four lists that gate `prev_logprobs`, `clip_ratio_c` and the rollout `mode` kwarg (same places as `GR00T_N1D7_UMI`) |
| FSDP wrapping | model class attrs + `configs/rl` | `_no_split_modules = ["GenLayer"]`; `_fsdp_wrap_embeddings = False` as RLinf's own `Cosmos3Policy` does, because `DomainAwareLinear` reads embedding weights directly; `wam_compile` off under RL |
| model config | `configs/model/umi_wam_v018.yaml` | `model_type umi_wam`, `model_path .../UMICore-v0.18.39/checkpoint-3000`, `embodiment_tag umi_bimanual_lpr1`, `obs_converter_type umi_bimanual`, `action_dim 20`, `num_action_chunks 16`, `denoising_steps 4`, `wam_und_cache_dir`, `rl_head_config` (flow_sde, `noise_level 0.3`, `action_noise_scale 0`, value head, `detach_obs_kv`) |
| run config | `configs/rl/laplacian_ppo_wam_v018_rigid.yaml` | clone of `laplacian_ppo_gr00t_umi_v0203_rigid_c.yaml` with the WAM model config; micro-batch 8 (half the parameters of N1.7) |
| und cache tool | `scripts/wam_build_und_cache.py` | moves `~/rlinf-runs/tools/wam_build_sim_cache.py` into the repo: builds a cache for a list of prompts from the readable Cosmos3-Edge und tower, keyed through `processor._formalize -> prompt_ids`, caption meta matching the checkpoint (`wrist_view`, 16 frames, 15 fps, 224 x 320) |
| checks | `scripts/wam_parity_check.py`, `scripts/wam_cond_parity.py`, `scripts/wam_rl_path_smoke.py` | WAM versions of the phase-0 scripts: eval path vs `Gr00tPolicy.get_action` (same generator), train-mode sampler at noise 0 vs `get_action`, rollout + replay ratio on one GPU |
| export | `scripts/export_hf_checkpoint.py` | drop the value head keys; copy `config.json`, processor files and the und cache (`save_pretrained(bundle_und_cache=True)` layout) so the fork's tools load the result |

Unchanged: the laplacian_gym env backend, both environments, the eval scripts (`gym_episode_video.py`
and `eval_queue.py` take `--model-cfg configs/model/umi_wam_v018.yaml`), the eval protocol.

## Phases

### W0. Load, parity, baseline (about 1 day)
1. Model config + `get_model` + eval-mode `predict_action_batch` only.
2. `wam_parity_check.py`: the same sim observation through `Gr00tPolicy.get_action` and through
   the wrapper's eval path, same `generator` seed. Verify: max |diff| = 0.0 on the decoded 14-dim
   action (the phase-0 criterion).
3. Baseline of v0.18.39@3000 in sim under the strict protocol: rigid 3 objects x 8 seeds
   (second-draw rule as in PHASE4), garment 8 seeds. Verify: numbers recorded in a `docs/PHASE5.md`
   next to the v0.20.3 baseline 33/144.

### W1. RL surface on one GPU, no Ray (about 2 days)
1. Train-mode sampler with chains and log-probs; `wam_cond_parity.py`: train mode at
   `noise_level 0` reproduces `get_action` to bf16 precision (the cond_add lesson of PHASE3).
2. `default_forward` replay; `wam_rl_path_smoke.py`: rollout then replay of the same chains with
   the same weights. Verify: `exp(logprobs - prev_logprobs)` = 1 within 1e-3, values finite,
   backward finite, peak memory at micro-batch 8 (expect about 14 GiB without checkpointing,
   measure with `wam_gradient_checkpointing`).
3. `sft_forward` wired (one batch from `umi_rl.offline.data` with the WAM processor in train mode);
   not exercised further in this plan.

### W2. PPO smoke through RLinf (about 1 day)
1. Fork commit on `integration` (special-case lists), submodule bump, run config.
2. `run_rl.sh laplacian_ppo_wam_v018_rigid runner.max_epochs=2 runner.save_interval=2` on GPUs
   4-7. Verify: finite actor/critic losses, `approx_kl` below 0.05, `clipped_ratio` near 1,
   checkpoint written, export loads through `gym_episode_video.py`.
3. Rollout and update timings recorded; adjust micro-batch and `detach_obs_kv` from them.

### W3. The real run (about 1 day of wall clock)
1. Rigid task, 120 epochs, the rigid_c recipe (32 envs, lr 1e-5, noise 0.3, 1 update epoch),
   checkpoints every 10, exports to `/home/thanh/models/umi-rl-wam-v018-rigid-step<N>`.
2. Same evaluation rules as PHASE4 (first draw 24, second draw at >= 8/24, third at >= 15/48,
   fourth for the 144-episode headline). Compare with the v0.18 baseline from W0 and with the
   v0.20.3 result (rigid_c step 60, 51/144).
3. Garment only if the W0 baseline shows two-pad contacts; v0.20.3 never produced one.

### W4 (optional). Offline training through RLinf
`UmiVlaSftWorker` with the WAM processor; needs the sentence closure of the dataset prompts in the
und cache (`sentence_closure.py`). Lower priority: the fork's own trainer already covers this.

## Risks and how they are handled

| Risk | Handling |
|---|---|
| und cache missing a prompt in a Ray worker | cache built up front for the env prompts; `has_sentence` asserted at `get_model`; `wam_und_live` kept off in RL (one und tower per worker is 1-2 GiB and a first-call stall) |
| checkpoints with `wam_attention: two_pass` (full pass A) | `get_model` raises; the RL loop mirrors the single-pass path only (v0.18.39 is `two_pass_block`) |
| `DomainAwareLinear` under FSDP | `_fsdp_wrap_embeddings = False` as in RLinf's `Cosmos3Policy`; verified by the W2 smoke |
| fork trains fp32 master weights under bf16 autocast, RL runs bf16 params | start as rigid_c did (bf16 FSDP, worked for N1.7); fall back to the FSDP2 fp32-param + autocast config of `umi_sft_smoke.yaml` if `approx_kl` drifts |
| `torch.compile` (`wam_compile: True`) | disabled for RL: dynamic batch and grad through the loop |
| frozen VAE encode per step | 3 views at 224 x 320, included in the measured 34 ms/sample |
| eval noise | +-2-3 per 24 episodes; the draw rules above |

Working rules as in PLAN.md: the fork is never edited in place (anything the WAM needs that is not
reachable from outside goes to `patches/`, then upstream), small commits, GPUs via
`component_placement`, everything GPU-side inside the container.
