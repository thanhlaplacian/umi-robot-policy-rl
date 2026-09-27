"""RLinf FSDP SFT worker for the company model: LeRobot + STEAM data through the fork's processor.

Differences from ``FSDPVlaSftWorker``: ``build_dataloader`` builds the SDK mixture dataset (which
shards itself by torch.distributed rank), and the epoch length comes from ``data.steps_per_epoch``
because the mixture dataset is an infinite iterable.
"""
from __future__ import annotations

from typing import Any

import torch

import umi_rl.model  # noqa: F401  registers gr00t_n1d7_umi inside every worker process
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


class UmiVlaSftWorker(FSDPVlaSftWorker):
    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        if eval_dataset:
            raise NotImplementedError("eval split not wired; use the fork's open-loop eval on saved weights")
        from omegaconf import OmegaConf

        from umi_rl.offline.data import OfflineDatasetSpec, build_dataset, build_processor

        d = self.cfg.data
        specs = []
        items = OmegaConf.to_container(data_paths, resolve=True) if not isinstance(data_paths, (list, str, dict)) else data_paths
        for item in (items if isinstance(items, list) else [items]):
            if isinstance(item, str):
                item = {"path": item}
            specs.append(OfflineDatasetSpec(
                path=item["path"], embodiment_tag=item.get("embodiment_tag", d.get("embodiment_tag", "umi_bimanual_v2")),
                mix_ratio=float(item.get("mix_ratio", 1.0)), steam_optimality=bool(item.get("steam_optimality", d.get("steam_optimality", False))),
                steam_optimality_threshold=float(item.get("steam_optimality_threshold", d.get("steam_optimality_threshold", 0.0))),
                eef_state_mask_prob=float(item.get("eef_state_mask_prob", d.get("eef_state_mask_prob", 0.0))),
                extra=dict(item.get("extra", {}))))
        ov = d.get("processor_overrides", None)
        overrides = OmegaConf.to_container(ov, resolve=True) if ov is not None else {}
        processor = build_processor(self.cfg.actor.model.model_path, **overrides)
        dataset = build_dataset(processor, specs, steam_label_version=d.get("steam_label_version"),
                                seed=int(self.cfg.actor.seed) + self._rank, num_shards_per_epoch=int(d.get("num_shards_per_epoch", 100000)),
                                video_backend=d.get("video_backend", "pyav"))
        try:
            from torchdata.stateful_dataloader import StatefulDataLoader as Loader
        except ImportError:  # pragma: no cover
            Loader = torch.utils.data.DataLoader
        loader = Loader(dataset, batch_size=int(self.cfg.actor.micro_batch_size), collate_fn=processor.collator,
                        num_workers=int(d.get("num_workers", 4)), pin_memory=False, persistent_workers=int(d.get("num_workers", 4)) > 0)
        self._steps_per_epoch = int(d.get("steps_per_epoch", 1000))
        return loader, {"specs": [s.__dict__ for s in specs]}

    def get_max_steps_per_epoch(self):
        return self._steps_per_epoch if self.data_loader is not None else 0
