"""Build training batches for the company model from LeRobot datasets with STEAM labels.

Reuses the fork's own machinery end to end (``Gr00tN1d7Processor`` in train mode,
``DatasetFactory`` -> ``ShardedMixtureDataset``, ``processor.collator``) so a batch here is the
same object the fork's SFT trainer feeds to ``Gr00tN1d7.forward``: ``state``, ``action``,
``action_mask``, ``embodiment_id``, VLM tokens/pixels, ``progress*`` and the STEAM
``optimality`` flag (1 = frame advantage >= threshold, 0 = sub-optimal, stamped 1 when a dataset
has no labels). RLinf's actor forward keeps exactly the keys the model needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class OfflineDatasetSpec:
    path: str
    embodiment_tag: str = "umi_bimanual_v2"
    mix_ratio: float = 1.0
    steam_optimality: bool = False
    steam_optimality_threshold: float = 0.0
    eef_state_mask_prob: float = 0.0
    extra: dict = field(default_factory=dict)  # any other SingleDatasetConfig field


def build_processor(model_path: str, **overrides):
    """The checkpoint's processor in TRAIN mode (augmentation, state dropout, STEAM stamping)."""
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    proc = Gr00tN1d7Processor.from_pretrained(model_path, **overrides)
    proc.train()
    return proc


def build_dataset(processor, specs: list[OfflineDatasetSpec], *, steam_label_version: str | None = None,
                  seed: int = 42, num_shards_per_epoch: int = int(1e5), video_backend: str = "pyav"):
    """ShardedMixtureDataset over ``specs`` (iterable; shard-by-shard sampling, one dataset per batch)."""
    from umi_data_sdk.core.data_config import DataConfig, SingleDatasetConfig
    from umi_data_sdk.dataset.factory import DatasetFactory

    if any(s.steam_optimality for s in specs) and steam_label_version is None:
        raise ValueError("steam_label_version is required when a spec sets steam_optimality")
    datasets = [SingleDatasetConfig(dataset_paths=[s.path], embodiment_tag=s.embodiment_tag, mix_ratio=s.mix_ratio,
                                    steam_optimality=s.steam_optimality, steam_optimality_threshold=s.steam_optimality_threshold,
                                    eef_state_mask_prob=s.eef_state_mask_prob, **s.extra) for s in specs]
    cfg = DataConfig(datasets=datasets, modality_configs=processor.modality_configs,
                     use_cumulative_action=bool(getattr(processor, "use_cumulative_action", False)),
                     seed=seed, num_shards_per_epoch=num_shards_per_epoch, steam_label_version=steam_label_version)
    if hasattr(cfg, "video_backend"):
        cfg.video_backend = video_backend
    train_ds, _ = DatasetFactory(cfg).build(processor)
    return train_ds


def build_loader(dataset, processor, batch_size: int, num_workers: int = 0):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=processor.collator,
                                       num_workers=num_workers, pin_memory=False)
