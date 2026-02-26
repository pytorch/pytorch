"""Utilities to aid in testing exported ONNX models."""

__all__ = ["assert_onnx_program"]

from torch.onnx._internal.exporter._testing import assert_onnx_program


assert_onnx_program.__module__ = "torch.onnx.testing"
