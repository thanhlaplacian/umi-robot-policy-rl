"""Register umi_rl's model type in EVERY interpreter of the venv, lazily.

RLinf builds models inside Ray worker processes that never import ``umi_rl``; ``get_model`` then
returns None for ``model_type: gr00t_n1d7_umi``. A ``umi_rl.pth`` file in site-packages imports
this module at interpreter start. It costs nothing until ``rlinf.models`` is imported, at which
point ``umi_rl.model`` (and its ``register_model`` call) runs right after RLinf's registry exists.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys

_TARGET = "rlinf.models"


class _RegisterAfterImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            import umi_rl.model  # noqa: F401  registers gr00t_n1d7_umi

        loader.exec_module = exec_module
        return spec


def install():
    if not any(isinstance(f, _RegisterAfterImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _RegisterAfterImport())


install()
