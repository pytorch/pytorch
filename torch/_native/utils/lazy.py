# Lazy module proxy: defers a heavy import to first ATTRIBUTE ACCESS rather than to the
# binding, which keeps `import torch` free of DSL runtimes while call sites still write
# `mod.attr`. Pair it with the TYPE_CHECKING-real / else-lazy idiom so static tooling still
# resolves attributes.
#
# torch/onnx/_internal/_lazy_import._LazyModule is the same ten lines, duplicated
# deliberately: this loads during native-op registration, and reaching into torch.onnx would
# put an onnx dependency on that path.

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
