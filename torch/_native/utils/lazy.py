# Lazy module proxy for torch._native: defers a heavy import to first ATTRIBUTE ACCESS rather than
# to the binding, which is what keeps `import torch` free of DSL runtimes like cutlass (the contract
# test_no_dsl_imports_after_import_torch enforces) while call sites still write `mod.attr`.
#
# torch/onnx/_internal/_lazy_import._LazyModule is the same 10 lines. Duplicated deliberately: this
# is imported during native-op registration, and reaching into torch.onnx would put an onnx
# dependency on that path. Do not deduplicate without moving one somewhere neutral.
#
# Use the TYPE_CHECKING-real / else-lazy idiom so static tooling still resolves attrs:
#
#     if TYPE_CHECKING:
#         from .._cutedsl import traits as T
#     else:
#         T = LazyModule("torch._native.ops._cutedsl.traits")

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
