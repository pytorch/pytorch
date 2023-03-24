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
        decomposition_table: The decomposition table for graph ops.
            Default is for torch ops, including aten and prim.
        op_level_debug: Whether to export the model with op level debug
            information with onnxruntime evaluator. op_level_debug is not supported
            when dynamic axes is on.
        enable_dynamic_axes: Whether to export the model with dynamic axes. This would set
            the shape of input and nodes all to dynamic by following symbolic fx graph.
            op_level_debug is not supported when dynamic axes is on.
    """

    opset_version: int = _constants.ONNX_DEFAULT_OPSET
    use_binary_format: bool = True
    op_level_debug: bool = False
    # NOTE(titaiwang): What would be the best arg name for this?
    enable_dynamic_axes: bool = True
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

        # NOTE(titaiwang): op_level_debug needs fixed shape to generate example inputs
        # for torch ops and ONNX ops, but in dynamic export, we don't have this info, so
        # op_level_debug would be forced to False if enable_dynamic_axes is True
        # https://github.com/microsoft/onnx-script/issues/393
        if self.enable_dynamic_axes and self.op_level_debug:
            raise RuntimeError(
                "op_level_debug and enable_dynamic_shape are mutually execusive. Please set only one of them to be True"
            )
