"""Backward compatibility module for torch.onnx.symbolic_opset9."""

from __future__ import annotations


__all__: list[str] = []

from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import *  # noqa: F401,F403
from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import (  # noqa: F401
    _prepare_onnx_paddings,
    _reshape_from_tensor,
    _slice,
    _var_mean,
)
