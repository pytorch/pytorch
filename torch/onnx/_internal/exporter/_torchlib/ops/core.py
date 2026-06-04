"""torch.ops.aten operators under the `core` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TCH001,TCH002

from __future__ import annotations

import operator
import string
from typing import Sequence

from onnxscript.onnx_opset import opset18 as op

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import TReal, TRealOrUInt8
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


aten = torch.ops.aten

_EINSUM_SYMBOLS = string.ascii_letters


def _get_einsum_symbol(dim: int) -> str:
    if dim >= len(_EINSUM_SYMBOLS):
        raise AssertionError(
            "ONNX export for aten._trilinear only supports up to 52 dimensions"
        )
    return _EINSUM_SYMBOLS[dim]


def _build_trilinear_subscript(total_dim: int, expanded_dims: Sequence[int]) -> str:
    expanded_dims_set = set(expanded_dims)
    return "".join(
        _get_einsum_symbol(dim)
        for dim in range(total_dim)
        if dim not in expanded_dims_set
    )


def _build_trilinear_equation(
    total_dim: int,
    expand1: Sequence[int],
    expand2: Sequence[int],
    expand3: Sequence[int],
    sumdim: Sequence[int],
) -> str:
    sumdim_set = set(sumdim)
    output_subscript = "".join(
        _get_einsum_symbol(dim)
        for dim in range(total_dim)
        if dim not in sumdim_set
    )
    return (
        f"{_build_trilinear_subscript(total_dim, expand1)},"
        f"{_build_trilinear_subscript(total_dim, expand2)},"
        f"{_build_trilinear_subscript(total_dim, expand3)}->{output_subscript}"
    )


@onnx_impl(aten._trilinear.default, trace_only=True)
def aten__trilinear(
    i1: TReal,
    i2: TReal,
    i3: TReal,
    expand1: Sequence[int],
    expand2: Sequence[int],
    expand3: Sequence[int],
    sumdim: Sequence[int],
    unroll_dim: int = 1,
) -> TReal:
    """_trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1) -> Tensor"""

    del unroll_dim

    total_dim = len(i1.shape) + len(expand1)
    equation = _build_trilinear_equation(
        total_dim, expand1, expand2, expand3, sumdim
    )
    return op.Einsum(i1, i2, i3, equation=equation)


@onnx_impl(aten.bilinear.default, trace_only=True)
def aten_bilinear(
    input1: TReal,
    input2: TReal,
    weight: TReal,
    bias: TReal | None = None,
) -> TReal:
    """bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias=None) -> Tensor"""

    batch_shape = op.Shape(input1, start=0, end=-1)
    input1_shape = op.Shape(input1, start=-1)
    input2_shape = op.Shape(input2, start=-1)
    output_shape = op.Shape(weight, start=0, end=1)
    neg_1 = op.Constant(value_ints=[-1])

    weight_permuted = op.Transpose(weight, perm=[1, 0, 2])
    weight_flat = op.Reshape(
        weight_permuted,
        op.Concat(input1_shape, op.Mul(output_shape, input2_shape), axis=0),
    )

    result = op.MatMul(input1, weight_flat)
    result = op.Reshape(
        result,
        op.Concat(batch_shape, output_shape, input2_shape, axis=0),
    )
    result = op.Squeeze(op.MatMul(result, op.Unsqueeze(input2, neg_1)), neg_1)

    if bias is not None:
        result = op.Add(result, bias)
    return result


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
