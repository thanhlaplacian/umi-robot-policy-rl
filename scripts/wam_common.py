"""Shared helpers of the WAM check scripts: load the umi_wam policy and build simulator observation batches."""
from __future__ import annotations

import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_CFG = os.path.join(ROOT, "configs/model/umi_wam_v018.yaml")
PROMPT = "Pick up the objects on the shelf and place them in the basket"
OBS_CACHE = os.path.expanduser("~/rlinf-runs/cache/wam_sim_obs.npz")


def load_model(model_cfg: str = DEFAULT_MODEL_CFG, ckpt: str | None = None, **overrides):
    import logging

    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
    from omegaconf import OmegaConf

    import umi_rl.model  # noqa: F401  registers umi_wam
    from rlinf.models import get_model

    cfg = OmegaConf.load(model_cfg)
    if ckpt:
        cfg.model_path = ckpt
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v)
    model = get_model(cfg).cuda()
    return model, cfg


def env_obs_from(obs: dict, prompt: str = PROMPT) -> dict:
    n = obs["states"].shape[0]
    return {"states": obs["states"].cpu(), "task_descriptions": [prompt] * n,
            "wrist_left_images": obs["cam_wrist_left"], "wrist_right_images": obs["cam_wrist_right"],
            "head_images": obs.get("cam_head")}


def sim_batches(model=None, n_envs: int = 4, advance: int = 16, refresh: bool = False) -> list[dict]:
    """Two batches of ``n_envs`` simulator observations (rigid pick-place, seeds 0..n-1): the reset frame and the
    frame after ``advance`` env steps of the policy's eval actions. Cached as npz so later checks skip the env."""
    if os.path.exists(OBS_CACHE) and not refresh:
        z = np.load(OBS_CACHE)
        out = []
        for i in range(int(z["n"])):
            out.append({"states": torch.from_numpy(z[f"states{i}"]), "task_descriptions": [PROMPT] * z[f"states{i}"].shape[0],
                        "wrist_left_images": torch.from_numpy(z[f"wl{i}"]), "wrist_right_images": torch.from_numpy(z[f"wr{i}"]),
                        "head_images": torch.from_numpy(z[f"hd{i}"])})
        return out
    from umi_rl.envs.laplacian.rl_env import LaplacianRLConfig, LaplacianRLEnv

    env = LaplacianRLEnv(LaplacianRLConfig(seed=0, max_episode_steps=240), num_envs=n_envs, device="cuda:0")
    obs = env.reset(seeds=list(range(n_envs)))
    batches = [env_obs_from(obs)]
    with torch.no_grad():
        actions, _ = model.predict_action_batch(env_obs=batches[0], mode="eval")
    actions = torch.as_tensor(np.asarray(actions), device="cuda:0")
    for s in range(min(advance, actions.shape[1])):
        obs, *_ = env.step(actions[:, s])
    batches.append(env_obs_from(obs))
    os.makedirs(os.path.dirname(OBS_CACHE), exist_ok=True)
    arrays = {"n": len(batches)}
    for i, b in enumerate(batches):
        arrays[f"states{i}"] = b["states"].numpy()
        arrays[f"wl{i}"] = b["wrist_left_images"].cpu().numpy()
        arrays[f"wr{i}"] = b["wrist_right_images"].cpu().numpy()
        arrays[f"hd{i}"] = b["head_images"].cpu().numpy()
    np.savez_compressed(OBS_CACHE, **arrays)
    return batches


def fork_obs_from(env_obs: dict, video_keys, state_keys) -> dict:
    """RLinf env batch -> the nested observation the fork's Gr00tPolicy takes."""
    from umi_rl import converters

    flat = converters.convert_umi_obs_to_gr00t_format(env_obs)
    return {"video": {k: flat[f"video.{k}"] for k in video_keys},
            "state": {k: flat[f"state.{k}"] for k in state_keys},
            "language": {converters.LANGUAGE_KEY: [[t] for t in flat[converters.LANGUAGE_KEY]]}}
