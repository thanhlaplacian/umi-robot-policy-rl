"""Import-time compatibility between the company fork of GR00T and RLinf.

RLinf's N1.7 wrapper imports ``gr00t.data.embodiment_tags.EmbodimentTag``. The company fork has no
``gr00t.data`` package: its embodiment tags (including ``umi_bimanual_v2``) live in
``umi_data_sdk.core.embodiment_tags``. Instead of editing RLinf or the fork, this module registers
an alias package ``gr00t.data`` whose ``embodiment_tags`` submodule *is* the SDK module, so both
RLinf and any upstream-style import resolve to the same enum object.

Call :func:`install` before importing anything under ``rlinf.models.embodiment.gr00t``.
"""
from __future__ import annotations

import importlib
import sys
import types


def install() -> None:
    if "gr00t.data.embodiment_tags" in sys.modules:
        return
    try:
        importlib.import_module("gr00t.data.embodiment_tags")
        return  # upstream Isaac-GR00T on the path: nothing to do
    except ModuleNotFoundError:
        pass
    import gr00t  # noqa: F401  (the fork's package must be importable)

    sdk_tags = importlib.import_module("umi_data_sdk.core.embodiment_tags")
    data_pkg = types.ModuleType("gr00t.data")
    data_pkg.__path__ = []  # mark as package so ``gr00t.data.embodiment_tags`` is a valid submodule
    data_pkg.embodiment_tags = sdk_tags
    sys.modules["gr00t.data"] = data_pkg
    sys.modules["gr00t.data.embodiment_tags"] = sdk_tags
    setattr(gr00t, "data", data_pkg)
