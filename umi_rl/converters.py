"""Observation / action converters between an RLinf env batch and the fork's GR00T processor.

Registered into RLinf's module-level registries (``OBS_CONVERSION`` and ``ACTION_CONVERSION_N1D7``
in ``rlinf.models.embodiment.gr00t.simulation_io``) under the key ``"umi_bimanual"``.

Contract (verified against ``Gr00tN1d7Processor.process_observation`` / ``decode_action`` in the
pinned fork, see docs/PHASE0.md):

* processor input: flat dict with ``video.<view>`` uint8 ``(B, T=1, H, W, 3)``, ``state.<key>``
  float32 ``(B, T=1, D)`` for the six state keys, and the language key as ``list[str]``;
* processor output (decoded): ``{key: (B, 16, D)}`` per-step EE-local deltas (xyz + axis-angle)
  with absolute grippers, i.e. exactly the dataset ``action`` convention.
"""
from __future__ import annotations

import numpy as np
import torch

CONVERTER_KEY = "umi_bimanual"

# Dataset / processor order: RIGHT arm first, then LEFT. 1+3+3+1+3+3 = 14 raw dims.
STATE_KEYS = (
    ("JOINT_GRIPPER_RIGHT", 1),
    ("FRAME_END_EFFECTOR_RIGHT_pos", 3),
    ("FRAME_END_EFFECTOR_RIGHT_rot", 3),
    ("JOINT_GRIPPER_LEFT", 1),
    ("FRAME_END_EFFECTOR_LEFT_pos", 3),
    ("FRAME_END_EFFECTOR_LEFT_rot", 3),
)
ACTION_KEYS = STATE_KEYS  # same six keys, same order, same widths
RAW_ACTION_DIM = sum(d for _, d in ACTION_KEYS)  # 14
LANGUAGE_KEY = "annotation.human.action.task_description"

# env_obs image key -> processor view name. Either the explicit keys or RLinf's real-env layout
# (``main_images`` + ``extra_view_images[:, i]``) may be used; the latter is mapped in this order.
VIEW_KEYS = (("wrist_left_images", "cam_wrist_left"), ("wrist_right_images", "cam_wrist_right"), ("head_images", "cam_head"))
REAL_ENV_VIEW_ORDER = ("cam_wrist_left", "cam_wrist_right", "cam_head")


def _to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _images_by_view(env_obs: dict) -> dict[str, np.ndarray]:
    views = {}
    for env_key, view in VIEW_KEYS:
        if env_obs.get(env_key) is not None:
            views[view] = _to_numpy(env_obs[env_key])
    if not views and env_obs.get("wrist_images") is not None:
        # RLinf sim layout used by the laplacian_gym backend: wrist_images [B, 2, H, W, 3] = (left, right),
        # main_images = cam_head (optional)
        w = _to_numpy(env_obs["wrist_images"])
        assert w.ndim == 5 and w.shape[1] == 2, f"wrist_images must be [B,2,H,W,3], got {w.shape}"
        views["cam_wrist_left"], views["cam_wrist_right"] = w[:, 0], w[:, 1]
        if env_obs.get("main_images") is not None:
            views["cam_head"] = _to_numpy(env_obs["main_images"])
    if not views and env_obs.get("main_images") is not None:
        views[REAL_ENV_VIEW_ORDER[0]] = _to_numpy(env_obs["main_images"])
        extra = env_obs.get("extra_view_images")
        if extra is not None:
            extra = _to_numpy(extra)  # (B, n_cam, H, W, 3)
            for i in range(extra.shape[1]):
                views[REAL_ENV_VIEW_ORDER[i + 1]] = extra[:, i]
    if "cam_wrist_left" not in views or "cam_wrist_right" not in views:
        raise KeyError(f"UMI observation needs both wrist views; got env keys {sorted(env_obs)}")
    return views


def convert_umi_obs_to_gr00t_format(env_obs: dict) -> dict:
    """RLinf env batch -> the flat modality dict the fork's processor consumes.

    ``env_obs``: ``states`` float ``(B, 14)`` in dataset order; images uint8 ``(B, H, W, 3)`` or
    ``(B, T, H, W, 3)`` under ``wrist_left_images`` / ``wrist_right_images`` / optional
    ``head_images`` (or RLinf's ``main_images`` + ``extra_view_images``); ``task_descriptions``
    ``list[str]``.
    """
    obs = {}
    for view, img in _images_by_view(env_obs).items():
        img = np.ascontiguousarray(img, dtype=np.uint8)
        if img.ndim == 4:
            img = img[:, None]  # add T=1
        assert img.ndim == 5 and img.shape[-1] == 3, f"{view}: expected (B,T,H,W,3), got {img.shape}"
        obs[f"video.{view}"] = img
    states = _to_numpy(env_obs["states"]).astype(np.float32)
    assert states.ndim == 2 and states.shape[1] == RAW_ACTION_DIM, f"states must be (B, 14), got {states.shape}"
    off = 0
    for key, dim in STATE_KEYS:
        obs[f"state.{key}"] = states[:, None, off : off + dim]  # (B, T=1, D)
        off += dim
    task = env_obs["task_descriptions"]
    obs[LANGUAGE_KEY] = list(task) if not isinstance(task, str) else [task]
    return obs


def convert_to_umi_action_n1d7(action_chunk: dict, chunk_size: int) -> np.ndarray:
    """Decoded processor output -> float32 ``(B, chunk_size, 14)`` per-step deltas, dataset order.

    Layout per step: [grip_R, dx dy dz (R), rotvec (R), grip_L, dx dy dz (L), rotvec (L)]; the
    deltas are EE-local, forward, one 15-fps step each (the same thing ``action`` stores in the
    LeRobot parquet), so the deployment stack can consume them unchanged.
    """
    parts = []
    for key, dim in ACTION_KEYS:
        v = np.asarray(action_chunk[key], dtype=np.float32)
        if v.ndim == 2:  # unbatched (chunk, D)
            v = v[None]
        assert v.shape[-1] == dim, f"{key}: expected width {dim}, got {v.shape}"
        parts.append(v[:, :chunk_size])
    out = np.concatenate(parts, axis=-1)
    assert out.shape[-1] == RAW_ACTION_DIM, out.shape
    return out


def register() -> None:
    from rlinf.models.embodiment.gr00t import simulation_io as sio

    sio.OBS_CONVERSION[CONVERTER_KEY] = convert_umi_obs_to_gr00t_format
    sio.ACTION_CONVERSION_N1D7[CONVERTER_KEY] = convert_to_umi_action_n1d7
