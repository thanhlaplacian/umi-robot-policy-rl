#!/usr/bin/env python3
"""Offline (SFT / advantage-weighted) training of the company GR00T N1.7 policy with RLinf's FSDP
SFT stack. A copy of RLinf's examples/sft/train_vla_sft.py that uses UmiVlaSftWorker.

usage: bash scripts/run_offline_sft.sh <config name under configs/offline>
"""
import json
import logging
import os

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

import umi_rl.model  # noqa: F401  registers gr00t_n1d7_umi before validate_cfg looks at model_type
from rlinf.config import validate_cfg
from rlinf.runners.sft_runner import SFTRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from umi_rl.offline.sft_worker import UmiVlaSftWorker

mp.set_start_method("spawn", force=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra.main(version_base="1.1", config_path=os.path.join(ROOT, "configs", "offline"), config_name="umi_sft_smoke")
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    logging.info(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    if cfg.actor.training_backend not in ("fsdp", "fsdp2"):
        raise ValueError(f"{cfg.actor.training_backend} backend is not supported")
    actor_group = UmiVlaSftWorker.create_group(cfg).launch(cluster, name=cfg.actor.group_name, placement_strategy=placement.get_strategy("actor"))
    runner = SFTRunner(cfg=cfg, actor=actor_group)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
