"""torch.ops.aten operators under the `core` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TCH001

from __future__ import annotations

import operator

import onnxscript
from onnxscript.onnx_opset import opset18 as op

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import (
    TReal,
    TRealOrUInt8,
    TRealUnlessInt16OrInt8,
)
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


aten = torch.ops.aten

# CumProd is introduced in ONNX opset 26, which onnxscript does not yet expose as
# a generated ``opsetXX`` module. Build the opset object directly so we can emit
# the op without depending on the generated module existing.
op26 = onnxscript.values.Opset("", 26)


@onnx_impl((aten.abs.default, operator.abs), trace_only=True)
def aten_abs(self: TRealOrUInt8) -> TRealOrUInt8:
    """abs(Tensor self) -> Tensor"""

    return op.Abs(self)


@onnx_impl(aten.abs.default, complex=True, trace_only=True)
def aten_abs_complex(self: TRealOrUInt8) -> TRealOrUInt8:
    """abs(Tensor self) -> Tensor"""

    return op.ReduceL2(self, [-1], keepdims=False)


@onnx_impl((aten.add.Tensor, aten.add.Scalar, operator.add), trace_only=True)
def aten_add(self: TReal, other: TReal, alpha: float = 1.0) -> TReal:
    """add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    if alpha != 1.0:
        alpha = op.CastLike(alpha, other)
        other = op.Mul(other, alpha)
    return op.Add(self, other)


@onnx_impl((aten.add.Tensor, aten.add.Scalar), trace_only=True, complex=True)
def aten_add_complex(self: TReal, other: TReal, alpha: float = 1.0) -> TReal:
    """add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""

    return aten_add(self, other, alpha=alpha)


@onnx_impl(aten.cumprod.default, trace_only=True, opset_introduced=26)
def aten_cumprod(
    self: TRealUnlessInt16OrInt8, dim: int, dtype: int = -1
) -> TRealUnlessInt16OrInt8:
    """cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"""

    if dtype != -1:
        self = op.Cast(self, to=dtype)
    if len(self.shape) == 0:
        # A scalar
        result = op.Identity(self)
    else:
        result = op26.CumProd(self, dim)
    return result
