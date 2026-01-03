"""Backward compatibility module for torch.onnx.symbolic_opset10."""

from __future__ import annotations


__all__: list[str] = []

from torch.onnx._internal.torchscript_exporter.symbolic_opset10 import *  # noqa: F401,F403
from torch.onnx._internal.torchscript_exporter.symbolic_opset10 import (  # noqa: F401
    _slice,
)
