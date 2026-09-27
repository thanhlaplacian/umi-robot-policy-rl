#!/usr/bin/env python3
"""Does batching two prompts of different length (so one gets padded) change the actions?

Runs each sample alone (no padding) and both together (RLinf right-pads with padding_value),
with GR00T_VLSA_APPLY_MASK unset and set. The company checkpoint's vl_self_attention ignores the
prompt mask unless the patch is active, so the unmasked run is expected to drift."""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def one_run(mask: str):
    env = dict(os.environ, GR00T_VLSA_APPLY_MASK=mask)
    out = subprocess.run([sys.executable, __file__, "--worker"], env=env, capture_output=True, text=True)
    lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    if not lines:
        print(out.stdout[-3000:], out.stderr[-3000:]); raise SystemExit("worker failed")
    return lines[-1]


def worker():
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
    import umi_rl.model  # noqa: F401
    from umi_rl import converters
    from rlinf.models import get_model
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from umi_data_sdk.core.embodiment_tags import EmbodimentTag
    from umi_data_sdk.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from umi_data_sdk.dataset.sharded_single_step_dataset import extract_step_data

    cfg = OmegaConf.load(os.path.join(ROOT, "configs/model/gr00t_n1d7_umi.yaml"))
    cfg.rl_head_config.padding_value = 0
    tag = EmbodimentTag(cfg.embodiment_tag)
    model = get_model(cfg); model.eval()
    proc = Gr00tN1d7Processor.from_pretrained(cfg.model_path)
    modality = proc.modality_configs[tag.value]
    samples = []
    for ds, ep, st, prompt in [("/data/dataset/26-W33-TELE2-rebot-batch1", 0, 40, None),
                               ("/data/umi_dataset/26-W36-UMI2-batch1", 3, 50, "Pick up the objects on the table and place them in the basket, then push the basket to the left edge of the table.")]:
        loader = LeRobotEpisodeLoader(ds, modality_configs=modality, video_backend="pyav")
        dp = extract_step_data(loader.get_episode(ep), st, modality, tag)
        views = [k for k in modality["video"].modality_keys if k in dp.images]
        o = {"states": torch.from_numpy(np.concatenate([np.asarray(dp.states[k], np.float32)[0] for k, _ in converters.STATE_KEYS])[None]),
             "task_descriptions": [prompt or dp.text]}
        env_key = {v: e for e, v in converters.VIEW_KEYS}
        for v in views:
            o[env_key[v]] = torch.from_numpy(np.asarray(dp.images[v])[0][None].astype(np.uint8))
        samples.append(o)
    # force the padded path: disable RLinf's eval grouping-by-prompt so the batch is really padded
    os.environ["RLINF_GR00T_GROUP_BY_PROMPT"] = "0"

    def run(obs):
        torch.manual_seed(0)
        with torch.no_grad():
            act, _ = model.predict_action_batch(env_obs=obs, mode="eval")
        return np.asarray(act, np.float32)
    alone = [run(o) for o in samples]
    both = run({k: (torch.cat([s[k] for s in samples]) if torch.is_tensor(samples[0][k]) else [s[k][0] for s in samples]) for k in samples[0]})
    tok = [len(model._modality_transform.processor.tokenizer(s["task_descriptions"][0])["input_ids"]) for s in samples]
    d = [float(np.abs(alone[i][0] - both[i]).max()) for i in range(2)]
    scale = float(np.mean([np.abs(a).mean() for a in alone]))
    print(f"RESULT mask={os.environ.get('GR00T_VLSA_APPLY_MASK','0')} prompt_tokens={tok} max|Δ| padded-vs-alone: short={d[0]:.3e} long={d[1]:.3e} (mean|a|={scale:.3e})")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        for m in ("0", "1"):
            print(one_run(m))
