"""torch.ops.aten operators under the `core` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# ruff: noqa: TCH001,TCH002
# flake8: noqa

from __future__ import annotations

import math

from onnxscript.onnx_opset import opset18 as op, opset20 as op20

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import TReal
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


_MATH_PI = math.pi


aten = torch.ops.aten


@onnx_impl(aten.gelu.default, trace_only=True)
def aten_gelu(self: TReal, approximate: str = "none") -> TReal:
    """gelu(Tensor self, *, str approximate='none') -> Tensor"""

    if approximate == "tanh":
        result = _aten_gelu_approximate_tanh(self)
    else:
        result = _aten_gelu_approximate_none(self)
    return result


@onnx_impl(aten.gelu.default, private=True)
def _aten_gelu_approximate_none(self: TReal) -> TReal:
    """gelu(Tensor self, *, str approximate='none') -> Tensor"""

    # GELU(x) = 0.5 * x * [1 + ERF(x/sqrt(2)]
    inner = op.Div(self, 1.4142135623730951)
    erf = op.Erf(inner)
    inner = op.Add(erf, 1)
    inner = op.Mul(0.5, inner)
    result = op.Mul(self, inner)
    return result


@onnx_impl(aten.gelu.default, private=True)
def _aten_gelu_approximate_tanh(self: TReal) -> TReal:
    """gelu(Tensor self, *, str approximate='none') -> Tensor"""

    # GELU(x) = 0.5 * x * {1 + Tanh[\sqrt(2/pi) * (x + 0.044715 * x^3)]}
    cubed = op.Pow(self, 3)
    inner = op.Mul(0.044715, cubed)
    inner = op.Add(self, inner)
    # Prefer explicit graph construction over precomputed constants for clarity.
    two_over_pi = op.CastLike(op.Div(2.0, _MATH_PI), self)
    inner = op.Mul(op.Sqrt(two_over_pi), inner)
    inner = op.Tanh(inner)
    inner = op.Add(inner, 1)
    inner = op.Mul(0.5, inner)
    result = op.Mul(self, inner)
    return result


@onnx_impl(aten.gelu.default, trace_only=True, opset_introduced=20)
def aten_gelu_opset20(
    self: TReal,
    approximate: str = "none",
) -> TReal:
    """gelu(Tensor self, *, bool approximate=False) -> Tensor"""
    return op20.Gelu(self, approximate=approximate)
