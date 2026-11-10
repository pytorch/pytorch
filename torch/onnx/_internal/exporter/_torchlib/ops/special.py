# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR special translations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


if TYPE_CHECKING:
    from torch.onnx._internal.exporter._ir_builder import TranslationContext


aten = torch.ops.aten
_MATH_PI = math.pi


@onnx_impl((aten.erf.default, aten.special_erf.default), opset_introduced=18)
def aten_special_erf(op: TranslationContext, self: ir.Value) -> ir.Value:
    """erf(Tensor self) -> Tensor"""
    return op.Erf(self)


@onnx_impl((aten.erfc.default, aten.special_erfc.default), opset_introduced=18)
def aten_special_erfc(op: TranslationContext, self: ir.Value) -> ir.Value:
    """erfc(Tensor self) -> Tensor"""
    return op.Sub(1, op.Erf(self))


@onnx_impl(aten.special_erfcx.default, opset_introduced=18)
def aten_special_erfcx(op: TranslationContext, self: ir.Value) -> ir.Value:
    """special_erfcx(Tensor self) -> Tensor"""
    return op.Mul(op.Exp(op.Pow(self, 2)), op.Sub(1, op.Erf(self)))


@onnx_impl((aten.expm1.default, aten.special_expm1.default), opset_introduced=18)
def aten_special_expm1(op: TranslationContext, self: ir.Value) -> ir.Value:
    """special_expm1(Tensor self) -> Tensor"""
    return op.Sub(op.Exp(self), 1)


@onnx_impl(
    (aten.log_softmax.int, aten.special_log_softmax.default),
    opset_introduced=18,
)
def aten_special_log_softmax(
    op: TranslationContext, self: ir.Value, dim: int, dtype: int = -1
) -> ir.Value:
    """special_log_softmax(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"""
    self_is_scalar = len(self.shape) == 0
    if self_is_scalar:
        self = op.Unsqueeze(self, [0])
    result = op.LogSoftmax(self, axis=dim)
    if dtype != -1:
        result = op.Cast(result, to=dtype)
    if self_is_scalar:  # squeeze to scalar due to input is scalar
        result = op.Squeeze(result)
    return result


@onnx_impl((aten.special_sinc.default, aten.sinc.default), opset_introduced=18)
def aten_special_sinc(op: TranslationContext, self: ir.Value) -> ir.Value:
    """special_sinc(Tensor self) -> Tensor"""
    # This computes the normalized sinc, where the input is multiplied by pi.
    # https://pytorch.org/docs/stable/special.html#torch.special.sinc
    pi_self = op.Mul(self, _MATH_PI)

    return op.Where(
        op.Equal(self, 0.0), op.CastLike(1, self), op.Div(op.Sin(pi_self), pi_self)
    )


@onnx_impl(
    (aten.xlogy.Tensor, aten.xlogy.Scalar_Self, aten.xlogy.Scalar_Other),
    opset_introduced=18,
)
def aten_special_xlogy(
    op: TranslationContext, self: ir.Value, other: ir.Value
) -> ir.Value:
    """special_xlogy(Tensor self, Tensor other) -> Tensor"""
    if not isinstance(self, ir.Value):
        self = op.CastLike(self, other)
    if not isinstance(other, ir.Value):
        other = op.CastLike(other, self)

    # https://pytorch.org/docs/stable/special.html#torch.special.xlogy
    # out := {
    #     NaN if other == NaN
    #     0 if self == 0
    #     self * log(other) otherwise
    # }

    nans = op.IsNaN(other)
    zeros = op.Equal(self, 0)
    xlogy = op.Mul(self, op.Log(other))
    xlogy_with_nans = op.Where(nans, other, xlogy)
    return op.Where(zeros, self, xlogy_with_nans)
