"""Utility to lazily import modules."""

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING


class _LazyModule:
    """Lazily import a module."""

    def __init__(self, module_name: str) -> None:
        self._name = module_name
        self._module: Any = None

    def __repr__(self) -> str:
        return f"<lazy module '{self._name}'>"

    def __getattr__(self, attr: str) -> object:
        if self._module is None:
            self._module = importlib.import_module(".", self._name)
        return getattr(self._module, attr)


# Import the following modules during type checking to enable code intelligence features,
# such as auto-completion in tools like pylance, even when these modules are not explicitly
# imported in user code.
# NOTE: Add additional used imports here.
if TYPE_CHECKING:
    import onnx
    import onnx_ir  # type: ignore[import-untyped]
    import onnxscript
    import onnxscript._framework_apis.torch_2_8 as onnxscript_apis

    onnxscript_ir = onnx_ir

else:
    onnx = _LazyModule("onnx")
    onnxscript = _LazyModule("onnxscript")
    onnxscript_ir = _LazyModule("onnx_ir")
    onnxscript_apis = _LazyModule("onnxscript._framework_apis.torch_2_8")
