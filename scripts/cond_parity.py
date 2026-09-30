#!/usr/bin/env python
"""Does RLinf's train-mode sampler (noise 0) reproduce the fork's eval ``get_action``?

Same observation, same initial noise (torch.manual_seed before each call). Before the
``rl_cond_add`` hook the RLinf path ran the DiT without the fork's embodiment/progress
condition rows, so the two differed; after it they must agree to bf16 precision.
  bash scripts/in_container.sh -g 0 python scripts/cond_parity.py [--ckpt ...]
"""
import argparse, os, sys, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data/models/UMICore-v0.16.10/checkpoint-50000")
    ap.add_argument("--model-cfg", default=os.path.join(ROOT, "configs/model/gr00t_n1d7_umi.yaml"))
    ap.add_argument("--dataset", default="/data/dataset/26-W33-TELE2-rebot-batch1")
    ap.add_argument("--no-hook", action="store_true", help="measure the gap with the hook removed")
    a = ap.parse_args()
    from omegaconf import OmegaConf
    import umi_rl.model  # noqa: registers the hook
    from umi_rl import converters
    from rlinf.models import get_model
    import gr00t.model  # noqa
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from umi_data_sdk.core.embodiment_tags import EmbodimentTag
    from umi_data_sdk.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from umi_data_sdk.dataset.sharded_single_step_dataset import extract_step_data
    if a.no_hook:
        from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import FlowMatchingActionHeadForRLActionPrediction as H
        del H.rl_cond_add
    cfg = OmegaConf.load(a.model_cfg); cfg.model_path = a.ckpt
    model = get_model(cfg); model.eval()
    head = model.action_head
    head.rl_config = dict(head.rl_config); head.rl_config["noise_level"] = 0.0
    tag = EmbodimentTag(cfg.embodiment_tag)
    proc = Gr00tN1d7Processor.from_pretrained(cfg.model_path)
    modality = proc.modality_configs[tag.value]
    loader = LeRobotEpisodeLoader(a.dataset, modality_configs=modality, video_backend="pyav")
    traj = loader.get_episode(0)
    dps = [extract_step_data(traj, s, modality, tag) for s in (0, 60)]
    views = [k for k in modality["video"].modality_keys if k in dps[0].images]
    env_obs = {"states": torch.from_numpy(np.stack([np.concatenate([np.asarray(dp.states[k], np.float32)[0] for k, _ in converters.STATE_KEYS]) for dp in dps])),
               "task_descriptions": [dp.text for dp in dps]}
    env_key = {v: e for e, v in converters.VIEW_KEYS}
    for v in views:
        env_obs[env_key[v]] = torch.from_numpy(np.stack([np.asarray(dp.images[v])[0] for dp in dps]).astype(np.uint8))
    with torch.no_grad():
        torch.manual_seed(0); a_eval, _ = model.predict_action_batch(env_obs=env_obs, mode="eval")
        torch.manual_seed(0); a_train, res = model.predict_action_batch(env_obs=env_obs, mode="train")
        torch.manual_seed(0); a_eval2, _ = model.predict_action_batch(env_obs=env_obs, mode="eval")
    a_eval, a_train, a_eval2 = (np.asarray(x) for x in (a_eval, a_train, a_eval2))
    print(f"hook={'off' if a.no_hook else 'on'}  eval-vs-eval repeat max|diff| = {np.abs(a_eval - a_eval2).max():.3e}")
    d = np.abs(a_eval - a_train)
    print(f"fork eval get_action vs RLinf train-mode sampler (noise 0): max|diff| = {d.max():.3e}  mean = {d.mean():.3e}  (pos cols m, rot rad, grip)")
    print("per-dim max:", np.round(d.reshape(-1, d.shape[-1]).max(0), 4).tolist())


if __name__ == "__main__":
    main()
