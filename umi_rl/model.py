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
from omegaconf import DictConfig, OmegaConf

from umi_rl import compat, converters, embodiment

MODEL_TYPE = "gr00t_n1d7_umi"


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


def register() -> None:
    from rlinf.models import register_model

    register_model(MODEL_TYPE, get_model, category="embodied", force=True)


register()
