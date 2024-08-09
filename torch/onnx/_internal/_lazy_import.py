"""Utility to lazily import modules."""

import importlib
from typing import TYPE_CHECKING

import onnxscript.ir.convenience


class _LazyModule:
    """Lazily import a module."""

    def __init__(self, module_name: str) -> None:
        self._name = module_name

    def __repr__(self) -> str:
        return f"<lazy module '{self._name}'>"

    def __getattr__(self, attr):
        module_attr = attr.split(".", 1)
        if len(module_attr) == 2:
            module, attr_ = module_attr
            return getattr(importlib.import_module(f".{module}", self._name), attr_)
        else:
            return getattr(importlib.import_module(".", self._name), attr)


# Import the following modules during type checking to enable code intelligence features,
# such as auto-completion in tools like pylance, even when these modules are not explicitly
# imported in user code.
# NOTE: Add additional use imports here.
if TYPE_CHECKING:
    import onnx
    import onnxscript
    import onnxscript.evaluator

    onnxscript_ir = onnxscript.ir
    onnxscript_ir_convenience = onnxscript.ir.convenience
    onnxscript_evaluator = onnxscript.evaluator
else:
    onnx = _LazyModule("onnx")
    onnxscript = _LazyModule("onnxscript")
    onnxscript_ir = _LazyModule("onnxscript.ir")
    onnxscript_ir_convenience = _LazyModule("onnxscript.ir.convenience")
    onnxscript_evaluator = _LazyModule("onnxscript.evaluator")
