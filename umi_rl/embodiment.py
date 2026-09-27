"""Embodiment tags the RL side accepts, resolved to the fork's enum."""
from __future__ import annotations

from umi_rl import compat

compat.install()

from umi_data_sdk.core.embodiment_tags import EmbodimentTag  # noqa: E402

EMBODIMENT_TAG_BY_CFG = {
    "umi_bimanual": EmbodimentTag.UMI_BIMANUAL,
    "umi_bimanual_v2": EmbodimentTag.UMI_BIMANUAL_V2,
    "umi_bimanual_t2": EmbodimentTag.UMI_BIMANUAL_T2,
    "umi_rb5": EmbodimentTag.UMI_RB5,
    "umi_rb5_t2": EmbodimentTag.UMI_RB5_T2,
    "oxe_droid_eef": EmbodimentTag.OXE_DROID_EEF,
    "new_embodiment": EmbodimentTag.NEW_EMBODIMENT,
}


def resolve(tag: str) -> EmbodimentTag:
    try:
        return EMBODIMENT_TAG_BY_CFG[tag]
    except KeyError:
        raise ValueError(f"unsupported embodiment tag {tag!r}; known: {sorted(EMBODIMENT_TAG_BY_CFG)}") from None
