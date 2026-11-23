# Defer heavy imports until attribute access so `import torch` stays DSL-free while call
# sites use `mod.attr`; use a real TYPE_CHECKING import for static tools. This duplicates
# torch.onnx's proxy to avoid adding an ONNX dependency to native-op registration.

from __future__ import annotations

import importlib
from typing import Any


class LazyModule:
    """A stand-in for a module that imports the real module on first attribute access."""

    __slots__ = ("_name", "_module")

    def __init__(self, module_name: str) -> None:
        self._name = module_name
        self._module: Any = None

    def __repr__(self) -> str:
        state = "loaded" if self._module is not None else "lazy"
        return f"<LazyModule {self._name!r} ({state})>"

    def __getattr__(self, attr: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self._name)
        return getattr(self._module, attr)
