"""Utility to lazily import modules."""
# mypy: allow-untyped-defs
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


class _LazyModule:
    """Lazily import a module."""

    def __init__(self, module_name: str) -> None:
        self._name = module_name
        self._module = None

    def __repr__(self) -> str:
        return f"<lazy module '{self._name}'>"

    def __getattr__(self, attr):
        if self._module is None:
            self._module = importlib.import_module(".", self._name)
        return getattr(self._module, attr)


# Import the following modules during type checking to enable code intelligence features,
# such as auto-completion in tools like pylance, even when these modules are not explicitly
# imported in user code.
# NOTE: Add additional used imports here.
if TYPE_CHECKING:
    import onnx
    import onnxscript
    import onnxscript.evaluator
    import onnxscript.ir.convenience

    onnxscript_ir = onnxscript.ir
    onnxscript_ir_convenience = onnxscript.ir.convenience
    onnxscript_evaluator = onnxscript.evaluator
else:
    onnx = _LazyModule("onnx")
    onnxscript = _LazyModule("onnxscript")
    onnxscript_ir = _LazyModule("onnxscript.ir")
    onnxscript_ir_convenience = _LazyModule("onnxscript.ir.convenience")
    onnxscript_evaluator = _LazyModule("onnxscript.evaluator")
