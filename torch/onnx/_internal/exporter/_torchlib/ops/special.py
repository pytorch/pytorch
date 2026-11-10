# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR special translations."""

from __future__ import annotations

import math

import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._ir_builder import current_builder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


aten = torch.ops.aten
_MATH_PI = math.pi


@onnx_impl((aten.erf.default, aten.special_erf.default), opset_introduced=18)
def aten_special_erf(self: ir.Value) -> ir.Value:
    """erf(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Erf", self)[0]


@onnx_impl((aten.erfc.default, aten.special_erfc.default), opset_introduced=18)
def aten_special_erfc(self: ir.Value) -> ir.Value:
    """erfc(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Sub", 1, builder.call("Erf", self)[0])[0]


@onnx_impl(aten.special_erfcx.default, opset_introduced=18)
def aten_special_erfcx(self: ir.Value) -> ir.Value:
    """special_erfcx(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call(
        "Mul",
        builder.call("Exp", builder.call("Pow", self, 2)[0])[0],
        builder.call("Sub", 1, builder.call("Erf", self)[0])[0],
    )[0]


@onnx_impl((aten.expm1.default, aten.special_expm1.default), opset_introduced=18)
def aten_special_expm1(self: ir.Value) -> ir.Value:
    """special_expm1(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Sub", builder.call("Exp", self)[0], 1)[0]


@onnx_impl(
    (aten.log_softmax.int, aten.special_log_softmax.default), opset_introduced=18
)
def aten_special_log_softmax(self: ir.Value, dim: int, dtype: int = -1) -> ir.Value:
    """special_log_softmax(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"""
    builder = current_builder()

    self_is_scalar = len(self.shape) == 0
    if self_is_scalar:
        self = builder.call("Unsqueeze", self, [0])[0]
    result = builder.call("LogSoftmax", self, axis=dim)[0]
    if dtype != -1:
        result = builder.call("Cast", result, to=dtype)[0]
    if self_is_scalar:  # squeeze to scalar due to input is scalar
        result = builder.call("Squeeze", result)[0]
    return result


@onnx_impl((aten.special_sinc.default, aten.sinc.default), opset_introduced=18)
def aten_special_sinc(self: ir.Value) -> ir.Value:
    """special_sinc(Tensor self) -> Tensor"""
    builder = current_builder()

    # This computes the normalized sinc, where the input is multiplied by pi.
    # https://pytorch.org/docs/stable/special.html#torch.special.sinc
    pi_self = builder.call("Mul", self, _MATH_PI)[0]

    return builder.call(
        "Where",
        builder.call("Equal", self, 0.0)[0],
        builder.call("CastLike", 1, self)[0],
        builder.call("Div", builder.call("Sin", pi_self)[0], pi_self)[0],
    )[0]


@onnx_impl(
    (aten.xlogy.Tensor, aten.xlogy.Scalar_Self, aten.xlogy.Scalar_Other),
    opset_introduced=18,
)
def aten_special_xlogy(self: ir.Value, other: ir.Value) -> ir.Value:
    """special_xlogy(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()
    if not isinstance(self, ir.Value):
        self = builder.call("CastLike", self, other)[0]
    if not isinstance(other, ir.Value):
        other = builder.call("CastLike", other, self)[0]

    # https://pytorch.org/docs/stable/special.html#torch.special.xlogy
    # out := {
    #     NaN if other == NaN
    #     0 if self == 0
    #     self * log(other) otherwise
    # }

    nans = builder.call("IsNaN", other)[0]
    zeros = builder.call("Equal", self, 0)[0]
    xlogy = builder.call("Mul", self, builder.call("Log", other)[0])[0]
    xlogy_with_nans = builder.call("Where", nans, other, xlogy)[0]
    return builder.call("Where", zeros, self, xlogy_with_nans)[0]
