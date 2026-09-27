"""Observation / action converters between an RLinf env batch and the fork's GR00T processor.

Registered into RLinf's module-level registries (``OBS_CONVERSION`` and ``ACTION_CONVERSION_N1D7``
in ``rlinf.models.embodiment.gr00t.simulation_io``) under the key ``"umi_bimanual"``. The exact
dict keys the fork's processor expects are filled in by :func:`umi_rl.model.register`; see
docs/PHASE0.md for the contract.
"""
from __future__ import annotations

import numpy as np
import torch

CONVERTER_KEY = "umi_bimanual"


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def convert_umi_obs_to_gr00t_format(env_obs: dict) -> dict:
    """RLinf env batch -> the modality dict the fork's Gr00tN1d7Processor consumes.

    Expected ``env_obs`` keys (batched, B first):
      ``wrist_left_images``, ``wrist_right_images``: uint8 [B, H, W, 3] (or [B, T, H, W, 3])
      ``head_images`` (optional): uint8 [B, H, W, 3]
      ``states``: float [B, 14] (gripper_R, eef_R xyz, eef_R axis-angle, gripper_L, eef_L xyz, eef_L axis-angle)
      ``task_descriptions``: list[str]
    """
    raise NotImplementedError("filled in once the processor input contract is verified (phase 0 step 2)")


def convert_to_umi_action_n1d7(action_chunk: dict, chunk_size: int) -> np.ndarray:
    """Decoded processor output -> [B, chunk_size, 20] array in the order the deployment stack expects."""
    raise NotImplementedError("filled in once the processor output contract is verified (phase 0 step 2)")


def register() -> None:
    from rlinf.models.embodiment.gr00t import simulation_io as sio

    sio.OBS_CONVERSION[CONVERTER_KEY] = convert_umi_obs_to_gr00t_format
    sio.ACTION_CONVERSION_N1D7[CONVERTER_KEY] = convert_to_umi_action_n1d7
