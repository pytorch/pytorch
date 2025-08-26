"""Backward compatibility module for torch.onnx.symbolic_opset10."""

__all__ = []

from torch.onnx._internal.torchscript_exporter.symbolic_opset10 import *  # noqa: F401,F403
from torch.onnx._internal.torchscript_exporter.symbolic_opset10 import (  # noqa: F401
    _slice,
)
