"""RLinf model builder for the company GR00T N1.7 checkpoints (model_type ``gr00t_n1d7_umi``).

A copy of ``rlinf.models.embodiment.gr00t.gr00t_n1d7.get_model`` with three differences:
  * embodiment tags resolve through the fork's SDK enum (``umi_bimanual_v2`` ...), see
    :mod:`umi_rl.embodiment`;
  * no monkey-patching of ``gr00t.data`` (the shim in :mod:`umi_rl.compat` already makes the SDK
    enum the one RLinf sees);
  * the UMI observation/action converters are registered before the wrapper looks them up.
Registered with RLinf's ``register_model`` so configs can say ``model_type: gr00t_n1d7_umi``.
"""
from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from rlinf.utils.logging import get_logger, OmegaConf

from umi_rl import compat, converters, embodiment

MODEL_TYPE = "gr00t_n1d7_umi"
# scalar stats from the fork's training forward that are worth a logger line
_SFT_LOGGED_STATS = {"action_loss", "progress_loss", "optimality_loss", "advantage_weight_sum", "flow_loss", "wm_loss", "flare_loss"}


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    compat.install()
    converters.register()
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from transformers import AutoConfig, AutoModel

    try:
        AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
        AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
    except ValueError:
        pass  # already registered in this process

    from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity

    UmiGr00tN1d7ForRL = _model_class()

    emb_tag = embodiment.resolve(cfg.embodiment_tag)
    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    config = Gr00tN1d7Config.from_pretrained(str(model_path))
    if cfg.get("action_dim") is not None:
        config.action_dim = cfg.action_dim
    # Offline-RL knobs the fork implements inside its training forward (see docs/PHASE1.md):
    # advantage_weight_suboptimal in [0, 1] down-weights frames whose STEAM optimality flag is 0
    # (1.0 = plain SFT, 0.0 = filtered BC). None keeps the checkpoint's own setting.
    for key in ("advantage_weight_suboptimal",):
        if cfg.get(key) is not None:
            setattr(config, key, cfg.get(key))
    backbone_model_path = OmegaConf.select(cfg, "backbone_model_path", default=None)
    model = UmiGr00tN1d7ForRL.from_pretrained(
        config=config,
        local_model_path=str(model_path),
        pretrained_model_name_or_path=str(model_path),
        backbone_model_path=backbone_model_path,
        torch_dtype=torch_dtype,
        embodiment_tag=emb_tag,
        denoising_steps=cfg.denoising_steps,
        output_action_chunks=cfg.num_action_chunks,
        obs_converter_type=cfg.obs_converter_type,
        rl_head_config=cfg.rl_head_config,
    )
    model.to(torch_dtype)
    # The fork samples the flow-matching time from a Beta whose parameters became bf16 with the
    # cast above; torch has no bf16 Dirichlet sampler. Keep the distribution in fp32 (the fork's
    # trainer holds fp32 weights under autocast, so this matches its numerics).
    bd = getattr(model.action_head, "beta_dist", None)
    if bd is not None and bd.concentration1.dtype != torch.float32:
        model.action_head.beta_dist = torch.distributions.Beta(bd.concentration1.float(), bd.concentration0.float())
    if cfg.rl_head_config.add_value_head and hasattr(model.action_head, "value_head"):
        model.action_head.value_head._init_weights()
    if cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)
    return model


_MODEL_CLS = None


def _model_class():
    """RLinf's N1.7 RL wrapper with the processor loaded by the fork's own ``from_pretrained``.

    RLinf builds ``Gr00tN1d7Processor(**processor_kwargs)`` straight from ``processor_config.json``;
    the fork's loader is the one that knows its legacy key aliases (``optional_views_last`` ->
    ``cam_head_last``), drops keys the pinned code no longer declares, and reads statistics /
    embodiment ids itself. Defined lazily so importing ``umi_rl.model`` stays cheap.
    """
    global _MODEL_CLS
    if _MODEL_CLS is not None:
        return _MODEL_CLS
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
    )

    class UmiGr00tN1d7ForRL(GR00T_N1_7_ForRLActionPrediction):
        def _prepare_rollout_observation(self, env_obs):
            """Same as RLinf's, minus the bf16 round-trip of the RAW state.

            RLinf rounds the raw state to bf16 before normalization to mimic a quirk of its
            LIBERO training data. The fork normalizes the float32 raw state and only then casts
            to bf16, in training and in deployment alike; keeping RLinf's hack moved the
            normalized state by up to one bf16 ulp and the decoded action by ~1e-3 (measured,
            scripts/parity_inputs.py). Dropping it makes the two paths bit-identical.
            """
            env_obs = dict(env_obs)
            states = env_obs["states"]
            env_obs["states"] = states.detach().cpu().float() if torch.is_tensor(states) else torch.as_tensor(states, dtype=torch.float32)
            observations = self.obs_convert_fn(env_obs)
            obs_copy = observations.copy()
            is_batch = self._check_state_is_batched(obs_copy)
            if not is_batch:
                from rlinf.models.embodiment.gr00t.utils import unsqueeze_dict_values

                obs_copy = unsqueeze_dict_values(obs_copy)
            obs_copy = self._coerce_observation_values_to_numpy(obs_copy)
            return observations, obs_copy, is_batch

        def _get_action_from_normalized_input(self, normalized_input):
            """Eval-path sampling exactly as the fork's deployment wrapper runs it.

            RLinf wraps ``get_action`` in ``torch.autocast(bf16)``; the fork's ``Gr00tPolicy``
            (and planner-vla) run the bf16 model under ``inference_mode`` only. The autocast
            changes the numerics of the fp32 pieces of the head and moved the decoded action by
            ~1e-3 (scripts/parity_check.py); without it the two paths agree bit for bit.
            """
            with torch.inference_mode():
                model_pred = self.get_action(normalized_input)
            return model_pred["action_pred"].float()

        def unapply_transforms(self, action_dict, state=None, **kw):
            """Decode with non-finite normalized actions replaced by 0 (a NaN physics world feeds a NaN
            observation to the policy; the fork's rot6d decode would otherwise raise in SVD and kill the rollout)."""
            a = action_dict.get("action")
            if torch.is_tensor(a) and not torch.isfinite(a).all():
                bad = (~torch.isfinite(a)).flatten(1).any(1).sum().item()
                get_logger().warning(f"non-finite normalized actions in {bad} sample(s); replaced by 0 before decode")
                action_dict = {**action_dict, "action": torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).clamp(-3.0, 3.0)}
            return super().unapply_transforms(action_dict, state=state, **kw)

        def forward(self, forward_type=None, **kwargs):
            from rlinf.models.embodiment.base_policy import ForwardType

            if forward_type is None or forward_type == ForwardType.DEFAULT:
                return super().forward(**kwargs)
            if forward_type == ForwardType.SFT:
                loss = self.sft_forward(kwargs["data"])
                return {"loss": loss, **{k: v for k, v in self.last_sft_stats.items() if k in _SFT_LOGGED_STATS}}
            raise NotImplementedError(f"forward_type {forward_type} not supported by gr00t_n1d7_umi")

        def sft_forward(self, data, **kwargs):
            """Supervised / offline loss on a batch from ``umi_rl.offline.data`` (RLinf SFT hook).

            Runs the fork's own training forward (``Gr00tN1d7.forward``), i.e. flow-matching
            velocity MSE plus the auxiliary heads the checkpoint config enables, with the STEAM
            ``optimality`` flag consumed exactly as in the fork's SFT trainer (CFGRL conditioning
            and/or ``advantage_weight_suboptimal`` when the model config sets them). Returns the
            scalar loss; extra scalar stats go to ``self.last_sft_stats``.
            """
            from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead

            # the fork's collator returns BatchFeature({"inputs": batch}); unwrap whatever container
            inputs = data["inputs"] if hasattr(data, "keys") and "inputs" in data else data
            inputs = dict(inputs.items())
            device = next(self.parameters()).device
            inputs = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            # RLinf's RL head overrides prepare_input/forward with the PPO replay signature
            # (chains, denoise_inds); call the fork's training versions on the same modules.
            backbone_inputs, action_inputs = Gr00tN1d7.prepare_input(self, inputs)
            action_inputs = Gr00tN1d7ActionHead.prepare_input(self.action_head, inputs)
            action_inputs = action_inputs.to(device) if hasattr(action_inputs, "to") else action_inputs
            backbone_outputs = self.backbone(backbone_inputs)
            out = Gr00tN1d7ActionHead.forward(self.action_head, backbone_outputs, action_inputs)
            self.last_sft_stats = {k: float(v) for k, v in out.items() if torch.is_tensor(v) and v.numel() == 1 and k != "loss"}
            return out["loss"]

        @staticmethod
        def _load_processor_from_dir(processor_dir: Path, *, backbone_model_path):
            loading_kwargs = {"trust_remote_code": True}
            if backbone_model_path is not None:
                loading_kwargs["local_files_only"] = True
            processor = Gr00tN1d7Processor.from_pretrained(
                str(processor_dir), transformers_loading_kwargs=loading_kwargs
            )
            return processor, getattr(processor, "modality_configs", None)

    _MODEL_CLS = UmiGr00tN1d7ForRL
    return _MODEL_CLS


def _rl_cond_add(self, vl_embs, state_features, embodiment_id, backbone_output):
    """The fork's DiT ``cond_add`` addend, rebuilt for RLinf's train-mode sampler.

    RLinf's ``sample_mean_var_val`` calls the DiT without ``cond_add`` (the upstream N1.7
    DiT has no such slot), which on the company fork silently drops the embodiment row,
    the progress-conditioning row and (v0.17+) the arm-active rows that the fork's own
    ``get_action`` (our eval path) sums in. Mirrors ``get_action_with_features``:
    memory -> embodiment -> progress (fed back from the progress head) -> arm-active
    (fed back from the arm-active head). Predictions are recomputed per call; they
    depend only on the VL features so they are identical across denoising steps.
    """
    base = backbone_output.get("mem_cond_add", None) if backbone_output is not None else None
    cond = self._embodiment_cond(base, embodiment_id, vl_embs.dtype)
    attn = getattr(backbone_output, "backbone_attention_mask", None)
    if getattr(self, "progress_cond_encoder", None) is not None:
        if getattr(self, "progress_head", None) is None:
            raise RuntimeError("use_progress_conditioning without a progress head; nothing to feed back")
        logits, ok = self._progress_logits(vl_embs, attn)
        pred = torch.where(ok, torch.sigmoid(logits), logits.new_full((), float("nan")))
        cond = self._progress_cond(cond, pred, ok, vl_embs.dtype)
    if getattr(self, "arm_active_cond_embedding", None) is not None:
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import arm_active_cond_value
        if getattr(self, "arm_active_head", None) is None:
            raise RuntimeError("use_arm_active_conditioning without an arm-active head; nothing to feed back")
        arm_logits, arm_ok = self._arm_active_logits(vl_embs, attn)
        arm_pred = torch.where(arm_ok.unsqueeze(-1), torch.sigmoid(arm_logits), arm_logits.new_full((), float("nan")))
        arm_value, arm_null = arm_active_cond_value(arm_pred, torch.isfinite(arm_pred))
        rows = self._arm_active_cond_rows(arm_value.device)
        cond = self._arm_active_cond(cond, arm_value, arm_null, rows, vl_embs.dtype)
    return cond


def register() -> None:
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import FlowMatchingActionHeadForRLActionPrediction as _Head
    _Head.rl_cond_add = _rl_cond_add
    from rlinf.models import register_model

    register_model(MODEL_TYPE, get_model, category="embodied", force=True)


register()
