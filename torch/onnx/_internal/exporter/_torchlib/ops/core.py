"""torch.ops.aten operators under the `core` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TCH001,TCH002

from __future__ import annotations

import operator

from onnxscript.onnx_opset import opset18 as op

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import TReal, TRealOrUInt8
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


@onnx_impl(aten.unbind.int, trace_only=True)
def aten_unbind(self: TReal, dim: int = 0) -> tuple[TReal, ...]:
    """unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]"""

    # Get the size of the dimension to unbind
    # At trace time, this should be a concrete integer value
    num_outputs = self.shape[dim]

    # Create individual slices along the specified dimension
    # Each slice extracts one element along dim, then we squeeze it out
    results = []
    for i in range(num_outputs):
        # Slice to get a single "slice" at position i along dim
        # We use Slice op: Slice(data, starts, ends, axes, steps)
        sliced = op.Slice(
            self,
            starts=op.Constant(value_ints=[i]),
            ends=op.Constant(value_ints=[i + 1]),
            axes=op.Constant(value_ints=[dim]),
        )
        # Squeeze to remove the dimension of size 1
        squeezed = op.Squeeze(sliced, axes=op.Constant(value_ints=[dim]))
        results.append(squeezed)

    return tuple(results)
