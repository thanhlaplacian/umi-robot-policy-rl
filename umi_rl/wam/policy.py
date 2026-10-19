"""RLinf policy for the UMICore WAM (``UmiWam``, v0.18), ``model_type: umi_wam``.

The WAM has no VLM and no DiT head: the prompt is an und K/V row from a cache, the observation is
encoded once into per-layer K/V (``_observation_kv``), and every Euler step runs the 28-layer
Cosmos3-Edge generation tower on the state + action tokens (``trunk.run``). RLinf's N1.7 RL head
cannot drive that, so this class reimplements its flow-SDE rollout and replay on the WAM's own
serving pieces.

Flow convention. RLinf: ``x_t = (1 - t) x0 + t x1`` (x0 noise, x1 clean), ``v = x1 - x0``. WAM:
``x = (1 - sigma) a + sigma eps``, ``v = eps - a``. Hence ``t = 1 - sigma`` and
``v_rlinf = -v_wam``; RLinf's flow-SDE mean/std (``sample_mean_var_val``) carry over per step with
``t`` and ``delta`` read from the checkpoint's own sigma schedule.

* eval mode: the fork's ``UmiWam.get_action`` unchanged (bit parity with ``Gr00tPolicy``,
  ``scripts/wam_parity_check.py``);
* train mode (rollout): the same Euler loop with SDE noise on one sampled denoising step; chains
  ``[B, steps+1, T, 20]`` (packed R|L layout, fp32), log-probs ``[B, steps, T, 20]``, value ``[B, 1]``;
* replay (actor): ``_observation_kv`` again (with grad unless ``detach_obs_kv``), one tower pass
  at the stored step, log-prob of the stored next state, value.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from umi_rl import compat, converters, embodiment

MODEL_TYPE = "umi_wam"
# RL bookkeeping keys the rollout adds next to the processor's model inputs
_RL_KEYS = ("chains", "denoise_inds", "rollout_batch_size")
_SFT_LOGGED_STATS = {"action_loss", "video_loss", "progress_loss", "arm_active_loss", "idm_loss"}


def _logprob(sample: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Gaussian log-density per element; ``sigma == 0`` (deterministic) elements contribute 0."""
    zero = sigma == 0
    s = torch.where(zero, torch.ones_like(sigma), sigma)
    lp = -torch.log(s) - 0.5 * np.log(2 * np.pi) - 0.5 * ((sample - mu) / s) ** 2
    return torch.where(zero, torch.zeros_like(lp), lp)


def _zero_nonfinite(t: torch.Tensor, what: str) -> torch.Tensor:
    if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
        from rlinf.utils.logging import get_logger

        bad = int((~torch.isfinite(t)).reshape(t.shape[0], -1).any(1).sum()) if t.dim() else 1
        get_logger().warning(f"umi_wam: non-finite {what} in {bad} sample(s); zeroed")
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


_MODEL_CLS = None


class _VaeOneFrame:
    """Encode one frame per VAE call inside the RL paths. Batched conv kernels round each frame differently by its
    position in the batch (2e-2 on the latents, amplified by the tower to a chunk-level log-ratio of ~0.2 when the
    actor replays shuffled rows); per-frame calls make rollout and replay agree bit for bit. Eval mode keeps the
    checkpoint's chunking, i.e. stays identical to deployment."""

    def __init__(self, config):
        self.config = config

    def __enter__(self):
        self.saved = self.config.wam_vae_chunk
        self.config.wam_vae_chunk = 1

    def __exit__(self, *exc):
        self.config.wam_vae_chunk = self.saved


def model_class():
    """``UmiWamForRL``, defined lazily so importing this module does not import the fork."""
    global _MODEL_CLS
    if _MODEL_CLS is not None:
        return _MODEL_CLS
    compat.install()
    import gr00t.model.umi_wam.wam_core as wc
    from gr00t.model.umi_wam.umi_wam import UmiWam
    from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
    from rlinf.models.embodiment.modules.value_head import ValueHead

    class UmiWamForRL(UmiWam, BasePolicy):
        _no_split_modules = ["GenLayer"]
        # DomainAwareLinear (action_proj_in/out) keeps per-domain weights in nn.Embedding tables and
        # indexes them itself; RLinf's Cosmos3Policy opts out of per-embedding FSDP units for the same reason.
        _fsdp_wrap_embeddings = False

        # ---- setup ------------------------------------------------------------------------------------
        def setup_rl(self, processor, embodiment_tag, rl_cfg: dict, num_action_chunks: int, value_dtype):
            self.processor = processor  # plain attribute: not a module, never in the state dict
            self.embodiment_tag = embodiment_tag
            self.rl_config = dict(rl_cfg)
            self.rl_config.setdefault("noise_method", "flow_sde")
            self.rl_config.setdefault("noise_level", 0.5)
            if self.rl_config["noise_method"] != "flow_sde":
                raise ValueError(f"umi_wam supports noise_method flow_sde only, got {self.rl_config['noise_method']}")
            if self.rl_config.get("joint_logprob", False):
                raise ValueError("umi_wam: joint_logprob is not implemented (one SDE step per chunk)")
            if self._pass_a_full():
                raise ValueError(
                    f"umi_wam RL needs a single-pass checkpoint (wam_attention block / two_pass_block); "
                    f"this one is {self.config.wam_attention!r}"
                )
            # RoPE frequencies stay fp32. FSDP's mixed precision casts buffers to buffer_dtype (bf16), and the WAM's
            # position ids reach ~1.5e4 (text_temporal_offset), so bf16 frequencies move every rotary angle: the actor's
            # replay then disagrees with the rollout by ~0.3 in chunk-level log-ratio before any update (W2 smoke).
            self.__dict__["_rope_inv_freq"] = self.trunk.rotary_emb.inv_freq.detach().float().clone()
            self.num_action_chunks = int(num_action_chunks)
            self.n_real = self.item_dim * self.n_items  # packed real action dims (20)
            mc = processor.get_modality_configs()[embodiment_tag.value]
            self._video_keys = list(mc["video"].modality_keys)
            self._state_keys = list(mc["state"].modality_keys)
            self.value_head = None
            if self.rl_config.get("add_value_head", False):
                self.value_head = ValueHead(
                    input_dim=self.spec.hidden_size + self.n_items * self.spec.action_dim,
                    hidden_sizes=(1024, 512, 256),
                    output_dim=1,
                    activation="relu",
                    bias_last=True,
                ).to(device=next(self.trunk.parameters()).device, dtype=value_dtype)

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            # The trunk checkpoints its own layers (torch.utils.checkpoint, non-reentrant).
            self.trunk.gradient_checkpointing = True

        def gradient_checkpointing_disable(self):
            self.trunk.gradient_checkpointing = False

        # ---- observation -> model inputs -------------------------------------------------------------
        def _collate(self, env_obs: dict) -> tuple[dict, dict]:
            """RLinf env batch -> (model inputs [B, ...], raw batched states for decode), exactly the
            per-observation processor path of the fork's ``Gr00tPolicy._get_action``."""
            from umi_data_sdk.core.types import MessageType, VLAStepData
            from gr00t.policy.gr00t_policy import _rec_to_dtype

            flat = converters.convert_umi_obs_to_gr00t_format(env_obs)  # float32 raw states, uint8 views
            b = len(flat[converters.LANGUAGE_KEY])
            processed, states = [], []
            for i in range(b):
                step = VLAStepData(
                    images={k: flat[f"video.{k}"][i] for k in self._video_keys},
                    states={k: flat[f"state.{k}"][i] for k in self._state_keys},
                    actions={},
                    text=flat[converters.LANGUAGE_KEY][i],
                    embodiment=self.embodiment_tag,
                )
                states.append(step.states)
                processed.append(self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}]))
            kw = {}
            dev = next(self.trunk.parameters()).device
            if getattr(self.processor.collator, "accepts_image_device", False) and dev.type == "cuda":
                kw["image_device"] = dev
            collated = _rec_to_dtype(self.processor.collator(processed, **kw), dtype=torch.bfloat16)
            inputs = dict(collated["inputs"].items())
            batched_states = {k: np.stack([s[k] for s in states], axis=0) for k in self._state_keys}
            return inputs, batched_states

        def _decode(self, packed20: torch.Tensor, batched_states: dict) -> np.ndarray:
            """Normalized packed chunk [B, T, 20] -> env actions [B, chunk, 14] (Gr00tPolicy's decode)."""
            a = packed20.detach().float().cpu()
            if not torch.isfinite(a).all():
                from rlinf.utils.logging import get_logger

                get_logger().warning("umi_wam: non-finite normalized actions; replaced by 0 before decode")
                a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).clamp(-3.0, 3.0)
            full = F.pad(a, (0, self.config.max_action_dim - a.shape[-1]))
            dec = self.processor.decode_action(full.numpy(), self.embodiment_tag, batched_states)
            dec = {k: v.astype(np.float32) for k, v in dec.items()}
            return converters.convert_to_umi_action_n1d7(dec, chunk_size=self.num_action_chunks)

        # ---- shared pieces of the sampler ------------------------------------------------------------
        def _context(self, inp: dict, dtype, detach_obs: bool = False):
            """Everything a denoising step needs that depends only on the observation."""
            dev = inp["wam_wrist"].device
            rope = self.trunk.rotary_emb
            if rope.inv_freq.dtype != torch.float32:
                rope.inv_freq = self.__dict__["_rope_inv_freq"].to(rope.inv_freq.device)
            cfg = self.config
            if detach_obs:
                with torch.no_grad():
                    prefix, prefix_valid, pos, b, h_o, o_real, spans = self._observation_kv(inp, dtype)
            else:
                prefix, prefix_valid, pos, b, h_o, o_real, spans = self._observation_kv(inp, dtype)
            preds, cond = self._serve_heads(h_o, o_real, None, b, dev, dtype, spans)
            t_a = cfg.action_horizon
            dom = self._domain(b, dev, inp)
            hand_s = torch.arange(self.n_items, device=dev).view(1, -1).expand(b, -1)
            e_s = self.trunk.embed_actions(inp["wam_state"].to(dtype), dom, hand_s)
            a_real = torch.zeros(1, 1, self.spec.action_dim, device=dev, dtype=dtype)
            a_real[..., : self.item_dim] = 1.0
            groups = torch.cat(
                [
                    torch.full((b, self.n_items), wc.GROUP_S, device=dev),
                    torch.full((b, self.n_items * t_a), wc.GROUP_A, device=dev),
                ],
                1,
            )
            return {
                "b": b, "dev": dev, "dtype": dtype, "t_a": t_a, "prefix": prefix,
                "mask": wc.build_gen_mask(groups, groups, prefix_valid),
                "position_ids": torch.cat([pos["S"], pos["A"]], dim=2),
                "e_s": e_s, "cond": cond, "dom": dom, "hand_a": self._hand_index(b, t_a, dev),
                "a_real": a_real, "h_o": h_o, "o_real": o_real, "preds": preds,
            }

        def _velocity(self, ctx: dict, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            """WAM velocity (eps - a) at per-sample sigma ``s`` [B] for action tokens ``x`` [B, nT, 64]."""
            b, dtype = ctx["b"], ctx["dtype"]
            sig_tok = s.view(-1, 1).expand(b, x.shape[1]) if s.dim() else s.expand(b, x.shape[1])
            e_a = self.trunk.embed_actions(x, ctx["dom"], ctx["hand_a"]) + self._action_temb(sig_tok, None, dtype)
            if ctx["cond"] is not None:
                e_a = e_a + ctx["cond"].unsqueeze(1)
            h = self.trunk.run(torch.cat([ctx["e_s"], e_a], 1), ctx["position_ids"], ctx["prefix"], ctx["mask"])
            return self.trunk.unembed_actions(h[:, self.n_items :], ctx["dom"]) * ctx["a_real"]

        def _sigmas(self, dev) -> torch.Tensor:
            return wc.shifted_sigmas(int(self.config.wam_infer_steps), float(self.config.wam_infer_shift), device=dev)

        def _noise_level(self, dev) -> torch.Tensor:
            if self.rl_config.get("noise_anneal"):
                start, end, steps = self.rl_config.get("noise_params", (0.0, 0.0, 100))
                lvl = start + (end - start) * min(getattr(self, "global_step", 0), steps) / steps
            else:
                lvl = self.rl_config.get("noise_level", 0.5)
            return torch.tensor(float(lvl), device=dev)

        def _sde_step(self, x, v_wam, sigmas, idx, a_real):
            """Mean and std of the flow-SDE transition from step ``idx`` [B] (RLinf's formula in t = 1 - sigma)."""
            t = 1.0 - sigmas.float()  # ascending 0 -> 1
            t_i = t[idx].view(-1, 1, 1)
            delta = (t[idx + 1] - t[idx]).view(-1, 1, 1)
            v_r = -v_wam.float()
            x0_pred = x - v_r * t_i
            x1_pred = x + v_r * (1.0 - t_i)
            safe_t = torch.where(t == 0, t[1], t)
            sig = (self._noise_level(x.device) * torch.sqrt((1.0 - t) / safe_t))[:-1]
            sig_i = sig[idx].view(-1, 1, 1)
            mean = x0_pred * (1.0 - t_i - delta - sig_i**2 * delta / (2.0 * (1.0 - t_i))) + x1_pred * (t_i + delta)
            std = (torch.sqrt(delta) * sig_i).expand_as(x) * a_real.float()
            return mean * a_real.float(), std

        def _pack20(self, tokens: torch.Tensor) -> torch.Tensor:
            return self._pack_hands(tokens)[..., : self.n_real]

        def _unpack20(self, packed: torch.Tensor) -> torch.Tensor:
            return self._hand_tokens(F.pad(packed, (0, self.config.max_action_dim - packed.shape[-1])))

        def _value(self, ctx: dict) -> torch.Tensor:
            """Value from the mean of the real O hidden states and the state tokens; [B]."""
            m = ctx["o_real"].unsqueeze(-1).to(ctx["h_o"].dtype)
            pooled = (ctx["h_o"] * m).sum(1) / m.sum(1).clamp_min(1.0)
            feat = torch.cat([pooled, ctx["state_flat"]], dim=-1)
            if self.rl_config.get("detach_critic_input", True):
                feat = feat.detach()
            vh_dtype = next(self.value_head.parameters()).dtype
            return self.value_head(feat.to(vh_dtype))[:, 0].float()

        # ---- rollout ---------------------------------------------------------------------------------
        @torch.no_grad()
        def _rl_sample(self, inputs: dict, mode: str):
            inp = self.prepare_input(inputs)
            dtype = self.trunk.proj_in.weight.dtype
            with _VaeOneFrame(self.config):
                ctx = self._context(inp, dtype)
            ctx["state_flat"] = inp["wam_state"].reshape(ctx["b"], -1)
            b, dev, a_real = ctx["b"], ctx["dev"], ctx["a_real"]
            sigmas = self._sigmas(dev)
            steps = sigmas.numel() - 1
            x = (torch.randn(b, self.n_items * ctx["t_a"], self.spec.action_dim, device=dev, dtype=dtype) * a_real).float()
            noise_idx = random.randint(0, steps - 1) if mode == "train" else -1
            chains, logps = [self._pack20(x)], []
            for i in range(steps):
                v = self._velocity(ctx, x, sigmas[i]).float()
                # every chain state is rounded to bf16 before it is scored: FSDP's mixed precision casts the stored
                # chains to bf16 on their way into the actor's forward, and the replay must see the same values
                if i == noise_idx:
                    idx = torch.full((b,), i, device=dev, dtype=torch.long)
                    mean, std = self._sde_step(x, v, sigmas, idx, a_real)
                    x = (mean + torch.randn_like(mean) * std).to(torch.bfloat16).float()
                    lp = _logprob(x, mean, std)
                else:
                    x = (x + (sigmas[i + 1] - sigmas[i]).float() * v).to(torch.bfloat16).float()
                    lp = torch.zeros_like(x)
                chains.append(self._pack20(x))
                logps.append(self._pack20(lp))
            chains = torch.stack(chains, 1)  # [B, steps+1, T, 20]
            # Only the noisy step has a non-zero log-prob; return it summed over steps, [B, chunk, 20], so the actor
            # uses its own fp32 copy (passed through the FSDP forward it would be cast to bf16).
            logps = torch.stack(logps, 1)[:, :, : self.num_action_chunks].sum(1)
            if self.value_head is not None:
                values = self._value(ctx)[:, None]
            else:
                values = torch.zeros(b, 1, device=dev)
            denoise_inds = torch.full((b, steps), noise_idx, device=dev, dtype=torch.long)
            return chains[:, -1], {
                "prev_logprobs": _zero_nonfinite(logps, "prev_logprobs"),
                "prev_values": _zero_nonfinite(values, "prev_values"),
                "forward_inputs": {
                    "chains": _zero_nonfinite(chains, "chains"),
                    "denoise_inds": denoise_inds,
                    "rollout_batch_size": torch.full((b,), b, dtype=torch.long),
                    **{k: v for k, v in inputs.items() if torch.is_tensor(v)},
                },
            }

        def predict_action_batch(self, env_obs, mode: Literal["train", "eval"] = "train", **kwargs):
            inputs, batched_states = self._collate(env_obs)
            if mode == "eval":
                with torch.inference_mode():
                    pred = self.get_action(inputs)  # the fork's serving sampler, unchanged
                packed = pred["action_pred"].float()[..., : self.n_real]
                result = {"prev_logprobs": None, "prev_values": None, "forward_inputs": {}}
            else:
                packed, result = self._rl_sample(inputs, mode)
            return self._decode(packed, batched_states), result

        # ---- actor replay ----------------------------------------------------------------------------
        def forward(self, forward_type=None, **kwargs):
            if forward_type is None or forward_type == ForwardType.DEFAULT:
                return self.default_forward(**kwargs)
            if forward_type == ForwardType.SFT:
                loss = self.sft_forward(kwargs["data"])
                return {"loss": loss, **{k: v for k, v in self.last_sft_stats.items() if k in _SFT_LOGGED_STATS}}
            raise NotImplementedError(f"forward_type {forward_type} not supported by umi_wam")

        def default_forward(
            self,
            forward_inputs: dict,
            compute_logprobs: bool = True,
            compute_entropy: bool = False,
            compute_values: bool = True,
            use_cache: bool = False,
            **kwargs,
        ) -> dict[str, Any]:
            inputs = {k: v for k, v in forward_inputs.items() if k not in _RL_KEYS and torch.is_tensor(v)}
            inp = self.prepare_input(inputs)
            dtype = self.trunk.proj_in.weight.dtype
            with _VaeOneFrame(self.config):
                ctx = self._context(inp, dtype, detach_obs=bool(self.rl_config.get("detach_obs_kv", False)))
            ctx["state_flat"] = inp["wam_state"].reshape(ctx["b"], -1)
            b, dev = ctx["b"], ctx["dev"]
            # GEMM kernels round by batch size: the replay reproduces the rollout's log-probs exactly only at the
            # rollout's batch size (env batch per rollout worker); set actor.micro_batch_size to it.
            rb = forward_inputs.get("rollout_batch_size")
            if rb is not None and int(rb.reshape(-1)[0]) != b and not getattr(self, "_warned_rb", False):
                from rlinf.utils.logging import get_logger

                get_logger().warning(f"umi_wam: replay micro-batch {b} != rollout batch {int(rb.reshape(-1)[0])}; "
                                     "log-ratios carry batch-size rounding (set actor.micro_batch_size to the rollout batch)")
                self._warned_rb = True
            chains = forward_inputs["chains"].to(dev).float()
            idx = forward_inputs["denoise_inds"].to(dev)[:, 0].long()
            if (idx < 0).any():
                raise ValueError("umi_wam replay got an eval-mode rollout sample (denoise index -1)")
            rows = torch.arange(b, device=dev)
            x_pre = self._unpack20(chains[rows, idx])
            x_next = self._unpack20(chains[rows, idx + 1])
            sigmas = self._sigmas(dev)
            v = self._velocity(ctx, x_pre, sigmas[idx]).float()
            mean, std = self._sde_step(x_pre, v, sigmas, idx, ctx["a_real"])
            logprobs = self._pack20(_logprob(x_next, mean, std))[:, : self.num_action_chunks]
            out = {"logprobs": logprobs.float(), "entropy": None}
            if compute_values and self.value_head is not None:
                out["values"] = self._value(ctx)
            else:
                out["values"] = torch.zeros(b, device=dev)
            for k in ("logprobs", "values", "prev_logprobs"):
                if k in out:
                    out[k] = _zero_nonfinite(out[k], k)
            return out

        # ---- offline (SFT) ---------------------------------------------------------------------------
        def sft_forward(self, data, **kwargs):
            """The fork's training loss (``UmiWam.forward``) on a batch from the WAM processor."""
            inputs = data["inputs"] if hasattr(data, "keys") and "inputs" in data else data
            out = UmiWam.forward(self, dict(inputs.items()))
            self.last_sft_stats = {k: float(v) for k, v in out.items() if torch.is_tensor(v) and v.numel() == 1 and k != "loss"}
            al, am = out.get("action_loss"), out.get("action_mask")
            if torch.is_tensor(al) and torch.is_tensor(am):  # per-element in the fork's output; log the masked mean
                self.last_sft_stats["action_loss"] = float(al.detach().float().sum() / am.float().sum().clamp_min(1.0))
            return out["loss"]

    _MODEL_CLS = UmiWamForRL
    return _MODEL_CLS


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    compat.install()
    converters.register()
    from gr00t.model.umi_wam.processing_umi_wam import UmiWamProcessor

    cls = model_class()
    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    cache = OmegaConf.select(cfg, "wam_und_cache_dir", default=None)
    kw = {"dtype": torch_dtype}
    if cache:
        kw["wam_und_cache_dir"] = str(cache)
    model = cls.from_pretrained(str(model_path), **kw)
    proc_kw = {"wam_und_cache_dir": str(cache)} if cache else {}
    processor = UmiWamProcessor.from_pretrained(str(model_path), **proc_kw)
    processor.eval()
    rl_cfg = OmegaConf.to_container(cfg.rl_head_config, resolve=True) if cfg.get("rl_head_config") is not None else {}
    model.setup_rl(
        processor,
        embodiment.resolve(cfg.embodiment_tag),
        rl_cfg,
        num_action_chunks=cfg.num_action_chunks,
        value_dtype=torch_dtype,
    )
    steps = cfg.get("denoising_steps")
    if steps is not None and int(steps) != int(model.config.wam_infer_steps):
        raise ValueError(f"denoising_steps {steps} != checkpoint wam_infer_steps {model.config.wam_infer_steps}")
    for prompt in cfg.get("required_prompts", None) or []:
        if not processor.has_sentence(prompt):
            raise ValueError(f"umi_wam: the und cache {cache or '(checkpoint)'} has no row for {prompt!r}")
    if rl_cfg.get("disable_dropout", True):
        from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity

        replace_dropout_with_identity(model)
    model.set_frozen_modules_to_eval_mode()
    return model


def register() -> None:
    from rlinf.models import register_model

    register_model(MODEL_TYPE, get_model, category="embodied", force=True)
