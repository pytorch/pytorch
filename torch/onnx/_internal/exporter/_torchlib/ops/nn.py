"""torch.ops.aten operators under the `core` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# ruff: noqa: TCH001,TCH002
# flake8: noqa

from __future__ import annotations

import math

from onnxscript.onnx_opset import opset20 as op20

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import TReal
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


aten = torch.ops.aten


@onnx_impl(aten.gelu.default, trace_only=True, opset_introduced=20)
def aten_gelu_opset20(
    self: TReal,
    approximate: str = "none",
) -> TReal:
    """gelu(Tensor self, *, bool approximate=False) -> Tensor"""
    return op20.Gelu(self, approximate=approximate)
