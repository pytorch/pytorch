"""torch.ops.aten operators under the `core` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# ruff: noqa: TCH001,TCH002
# flake8: noqa

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


@onnx_impl((aten.grid_sampler.default), trace_only=True)
def aten_grid_sampler(
    input: TReal,
    grid: TReal,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> TReal:
    """grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor"""

    inter_mode_options = ("bilinear", "nearest", "bicubic")
    inter_mode_str = inter_mode_options[interpolation_mode]

    padding_mode_options = ("zeros", "border", "reflection")
    padding_mode_str = padding_mode_options[padding_mode]

    # Only one onnx Op so don't put into private function
    return op.GridSample(
        input,
        grid,
        align_corners=align_corners,
        mode=inter_mode_str,
        padding_mode=padding_mode_str,
    )


@onnx_impl((aten.grid_sampler.default), trace_only=True, opset_introduced=20)
def aten_grid_sampler(
    input: TReal,
    grid: TReal,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> TReal:
    """grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor"""

    inter_mode_options = ("bilinear", "nearest", "bicubic")
    inter_mode_str = inter_mode_options[interpolation_mode]
    if inter_mode_str == "bilinear":
        inter_mode_str = "linear"
    elif inter_mode_str == "bicubic":
        inter_mode_str = "cubic"

    padding_mode_options = ("zeros", "border", "reflection")
    padding_mode_str = padding_mode_options[padding_mode]

    # Only one onnx Op so don't put into private function
    from onnxscript.onnx_opset import opset20 as op20
    return op20.GridSample(
        input,
        grid,
        align_corners=align_corners,
        mode=inter_mode_str,
        padding_mode=padding_mode_str,
    )
