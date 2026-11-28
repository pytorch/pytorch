"""torch.ops.aten operators under the `core` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TCH001

from __future__ import annotations

import math
import operator

from onnxscript.onnx_opset import opset18 as op

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import (
    TFloat,
    TRealOrUInt8,
)
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


aten = torch.ops.aten


@onnx_impl((aten.abs.default, operator.abs), trace_only=True)
def aten_abs(self: TRealOrUInt8) -> TRealOrUInt8:
    """abs(Tensor self) -> Tensor"""

    return op.Abs(self)


@onnx_impl(aten.abs.default, complex=True, trace_only=True)
def aten_abs_complex(self: TRealOrUInt8) -> TRealOrUInt8:
    """abs(Tensor self) -> Tensor"""

    return op.ReduceL2(self, [-1], keepdims=False)


@onnx_impl(aten.logaddexp.default, trace_only=True)
def aten_logaddexp(self: TFloat, other: TFloat) -> TFloat:
    """logaddexp(Tensor self, Tensor other) -> Tensor"""

    # max(a, b) + log(1 + exp(-|a - b|)) does not overflow like log(exp(a) + exp(b)).
    # a == b is handled separately because a - b is NaN when both are +-inf.
    one = op.CastLike(1.0, self)
    diff = op.Abs(op.Sub(self, other))
    result = op.Add(op.Max(self, other), op.Log(op.Add(one, op.Exp(op.Neg(diff)))))
    return op.Where(
        op.Equal(self, other), op.Add(self, op.CastLike(math.log(2.0), self)), result
    )


@onnx_impl(aten.logaddexp2.default, trace_only=True)
def aten_logaddexp2(self: TFloat, other: TFloat) -> TFloat:
    """logaddexp2(Tensor self, Tensor other) -> Tensor"""

    # Same as aten_logaddexp, in base 2.
    one = op.CastLike(1.0, self)
    ln2 = op.CastLike(math.log(2.0), self)
    diff = op.Abs(op.Sub(self, other))
    log2_term = op.Div(op.Log(op.Add(one, op.Exp(op.Mul(op.Neg(diff), ln2)))), ln2)
    result = op.Add(op.Max(self, other), log2_term)
    return op.Where(op.Equal(self, other), op.Add(self, one), result)
