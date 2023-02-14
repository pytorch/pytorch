"""Options for FX exporter."""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict

import torch
from torch.onnx import _constants
from torch.onnx._internal.fx import function_dispatcher


@dataclasses.dataclass
class ExportOptions:
    """Options for FX-ONNX export.
    Attributes:
        opset_version: The export ONNX version.
        use_binary_format: Whether to Return ModelProto in binary format.
        decomposition_table: The decomposition table for graph ops. Default is for torch ops, including aten and prim.
        op_level_debug: Whether to export the model with op level debug information with onnxruntime evaluator.
    """

    opset_version: int = _constants.ONNX_DEFAULT_OPSET
    use_binary_format: bool = True
    op_level_debug: bool = False
    decomposition_table: Dict[torch._ops.OpOverload, Callable] = dataclasses.field(
        default_factory=lambda: function_dispatcher._ONNX_FRIENDLY_DECOMPOSITION_TABLE
    )

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                if value is not None:
                    setattr(self, key, value)
            else:
                raise KeyError(f"ExportOptions has no attribute {key}")
