"""torch.ops.aten operators under the `core` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# ruff: noqa: TCH001,TCH002
# flake8: noqa: B950

from __future__ import annotations

from typing import Optional

from onnxscript.onnx_opset import opset20 as op20, opset21 as op21

import torch
from torch.onnx._internal._lazy_import import onnxscript_ir as ir
from torch.onnx._internal.exporter._torchlib._tensor_typing import TFloat, TReal
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


aten = torch.ops.aten


@onnx_impl(aten.gelu.default, trace_only=True, opset_introduced=20)
def aten_gelu_opset20(
    self: TReal,
    approximate: str = "none",
) -> TReal:
    """gelu(Tensor self, *, bool approximate=False) -> Tensor"""
    return op20.Gelu(self, approximate=approximate)


@onnx_impl(aten.group_norm.default, trace_only=True, opset_introduced=21)
def aten_group_norm(
    input: TFloat,
    num_groups: int,
    weight: Optional[TFloat] = None,
    bias: Optional[TFloat] = None,
    eps: float = 1e-05,
    cudnn_enabled: bool = True,
) -> TFloat:
    """group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor"""

    c = op21.Shape(input, start=1, end=2)
    if weight is None:
        weight = op21.ConstantOfShape(c, value=ir.tensor(1.0, dtype=input.dtype))
    if bias is None:
        bias = op21.ConstantOfShape(c, value=ir.tensor(0.0, dtype=input.dtype))
    return op21.GroupNormalization(
        input, weight, bias, epsilon=eps, num_groups=num_groups
    )
