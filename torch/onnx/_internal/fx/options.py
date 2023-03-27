"""Options for FX exporter."""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict

import torch
import torch.fx
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
        static_reference_graph: torch.fx.graph object with real shape information to be
            used as a reference in op_level_debug mode.
    """

    opset_version: int = _constants.ONNX_DEFAULT_OPSET
    use_binary_format: bool = True
    op_level_debug: bool = False
    # NOTE(titaiwang): What would be the best arg name for this?
    enable_dynamic_axes: bool = True
    static_reference_graph: torch.fx.Graph = dataclasses.field(
        default_factory=torch.fx.Graph
    )
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
