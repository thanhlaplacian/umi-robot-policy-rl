# RLinf env backend for laplacian-gym (company simulator) driven by umi_rl.envs.laplacian.
# Mirrors rlinf/envs/sim/maniskill/maniskill_env.py: in-process batched GPU sim, group-based
# reset ids, chunk_step with last-column termination collapse, auto-reset bookkeeping.
from __future__ import annotations

from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf



def torch_clone_dict(d):
    """Deep-clone a (nested) dict of tensors; non-tensors are copied by reference."""
    if isinstance(d, dict):
        return {k: torch_clone_dict(v) for k, v in d.items()}
    return d.clone() if torch.is_tensor(d) else d


class LaplacianGymEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics=True):
        from umi_rl.envs.laplacian.rl_env import LaplacianRLConfig, LaplacianRLEnv

        self.cfg = cfg
        self.seed = cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.reward_scale = float(cfg.get("reward_scale", 1.0))
        self.ignore_terminations = cfg.ignore_terminations
        self.group_size = cfg.group_size
        self.num_group = num_envs // cfg.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.record_metrics = record_metrics
        self.num_envs = num_envs
        self.device = torch.device("cuda:0")  # RLinf remaps CUDA_VISIBLE_DEVICES per worker
        params = OmegaConf.to_container(cfg.init_params, resolve=True) if cfg.get("init_params") is not None else {}
        params.setdefault("seed", self.seed)
        params.setdefault("max_episode_steps", cfg.max_episode_steps)
        task = params.pop("task", "pick_place")
        if task == "garment":
            from umi_rl.envs.laplacian.garment_env import GarmentRLConfig, GarmentRLEnv
            prompt = params.pop("prompt", None)
            self.env = GarmentRLEnv(GarmentRLConfig(**params, prompt=prompt), num_envs=num_envs, device="cuda:0")
            self.prompt = self.env.cfg.prompt
        else:
            self.prompt = params.pop("prompt", "Pick up the objects on the shelf and place them in the basket")
            self.env = LaplacianRLEnv(LaplacianRLConfig(**params), num_envs=num_envs, device="cuda:0")
        self.total_num_group_envs = int(cfg.get("total_num_group_envs", 10_000))  # seed pool for episodes
        self._elapsed = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.prev_step_reward = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self._is_start = True
        self._init_metrics()
        self._init_reset_state_ids()

    # ---- bookkeeping mirrored from ManiskillEnv
    @property
    def elapsed_steps(self):
        return self._elapsed

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def instruction(self):
        return [self.prompt] * self.num_envs

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        ids = torch.randint(0, self.total_num_group_envs, (self.num_group,), generator=self._generator)
        self.reset_state_ids = ids.repeat_interleave(self.group_size).to(self.device)

    _TERM_KEYS = ("grasp", "lifted", "aligned", "stage", "n_in", "idle_pen", "disturb_pen", "orient")

    def _init_metrics(self):
        self.success_once = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.fail_once = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.returns = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._term_sum = {k: torch.zeros(self.num_envs, device=self.device) for k in self._TERM_KEYS}
        self._term_max = {k: torch.zeros(self.num_envs, device=self.device) for k in ("stage", "n_in")}

    def _reset_metrics(self, env_idx=None):
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if env_idx is not None:
            mask[:] = False
            mask[env_idx] = True
        self.prev_step_reward[mask] = 0.0
        self.success_once[mask] = False
        self.fail_once[mask] = False
        self.returns[mask] = 0.0
        self._elapsed[mask] = 0
        for v in self._term_sum.values():
            v[mask] = 0.0
        for v in self._term_max.values():
            v[mask] = 0.0

    def _record_metrics(self, step_reward, infos):
        self.returns += step_reward
        self.success_once = self.success_once | infos["success"]
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"]
        terms = infos.get("reward_terms", {})
        for k in self._TERM_KEYS:
            if k in terms:
                self._term_sum[k] += terms[k].float()
        for k in self._term_max:
            if k in terms:
                self._term_max[k] = torch.maximum(self._term_max[k], terms[k].float())
        ep = {"success_once": self.success_once.clone(), "fail_once": self.fail_once.clone(), "return": self.returns.clone(),
              "episode_len": self._elapsed.clone()}
        ep["reward"] = ep["return"] / ep["episode_len"].clamp_min(1)
        L = self._elapsed.clamp_min(1).float()
        ep["grasp_rate"] = self._term_sum["grasp"] / L
        ep["lift_rate"] = self._term_sum["lifted"] / L
        ep["align_rate"] = self._term_sum["aligned"] / L
        ep["orient_mean"] = self._term_sum["orient"] / L
        ep["stage_mean"] = self._term_sum["stage"] / L
        ep["stage_max"] = self._term_max["stage"].clone()
        ep["objects_in_basket_max"] = self._term_max["n_in"].clone()
        ep["idle_pen_mean"] = self._term_sum["idle_pen"] / L
        ep["disturb_pen_mean"] = self._term_sum["disturb_pen"] / L
        infos["episode"] = ep
        return infos

    # ---- observation contract: main = head cam (or None), wrist_images [N,2,H,W,3] (left, right)
    def _wrap_obs(self, obs):
        out = {"states": obs["states"], "task_descriptions": self.instruction}
        if "cam_wrist_left" in obs:
            out["wrist_images"] = torch.stack([obs["cam_wrist_left"], obs["cam_wrist_right"]], 1)
        if "cam_head" in obs:
            out["main_images"] = obs["cam_head"]
        return out

    def reset(self, *, seed=None, options: Optional[dict] = None):
        options = options or {}
        env_idx = options.get("env_idx")
        mask = None
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True
        if self.use_fixed_reset_state_ids:
            ids = options.get("episode_id", self.reset_state_ids)
            seeds = (self.reset_state_ids if mask is None else self.reset_state_ids.clone().index_put_((env_idx,), ids)).tolist()
        else:
            seeds = None
        obs = self.env.reset(mask, seeds=seeds)
        self._reset_metrics(env_idx)
        return self._wrap_obs(obs), {}

    def _calc_step_reward(self, reward):
        reward = reward * self.reward_scale
        if self.use_rel_reward:
            out = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return out
        return reward

    def step(self, actions: Union[np.ndarray, torch.Tensor], auto_reset=True):
        actions = torch.as_tensor(np.asarray(actions) if not torch.is_tensor(actions) else actions, device=self.device, dtype=torch.float32)
        obs, reward, terminations, truncations, info = self.env.step(actions)
        self._elapsed += 1
        infos = {"success": info["success"], "fail": info.get("fail", torch.zeros_like(info["success"])), "reward_terms": info.get("reward_terms", {})}
        step_reward = self._calc_step_reward(reward)
        infos = self._record_metrics(step_reward, infos)
        if self.ignore_terminations:
            terminations = torch.zeros_like(terminations)
            infos["episode"]["success_at_end"] = infos["success"].clone()
        dones = terminations | truncations
        extracted = self._wrap_obs(obs)
        if dones.any() and auto_reset and self.auto_reset:
            extracted, infos = self._handle_auto_reset(dones, extracted, infos)
        return extracted, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        chunk_size = chunk_actions.shape[1]
        obs_list, infos_list, rewards, terms, truncs = [], [], [], [], []
        for i in range(chunk_size):
            o, r, te, tr, inf = self.step(chunk_actions[:, i], auto_reset=False)
            obs_list.append(o); infos_list.append(inf); rewards.append(r); terms.append(te); truncs.append(tr)
        rewards = torch.stack(rewards, 1); terms = torch.stack(terms, 1); truncs = torch.stack(truncs, 1)
        past_t, past_u = terms.any(1), truncs.any(1)
        past_dones = past_t | past_u
        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(past_dones, obs_list[-1], infos_list[-1])
        chunk_t = torch.zeros_like(terms); chunk_t[:, -1] = past_t
        chunk_u = torch.zeros_like(truncs); chunk_u[:, -1] = past_u
        return obs_list, rewards, chunk_t, chunk_u, infos_list

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = torch_clone_dict(extracted_obs)
        final_info = torch_clone_dict(infos)
        env_idx = torch.arange(self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        if self.use_fixed_reset_state_ids:
            options["episode_id"] = self.reset_state_ids[env_idx]
        extracted_obs, infos = self.reset(options=options)
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    # ---- video support for RecordVideo
    def capture_image(self):
        frames = self.env.render_all()
        img = frames.get("cam_wrist_right", next(iter(frames.values())))
        return img.cpu().numpy()

    def render(self, info=None, reward=None):
        return self.capture_image()

    def close(self):
        pass
