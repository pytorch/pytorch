"""torch.ops.aten operators under the `core` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TCH001

from __future__ import annotations

import operator

import torch
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.onnx._internal.exporter._ir_builder import current_builder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


aten = torch.ops.aten


@onnx_impl((aten.abs.default, operator.abs))
def aten_abs(self: ir.Value) -> ir.Value:
    """abs(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Abs", self)[0]


@onnx_impl(aten.abs.default, complex=True)
def aten_abs_complex(self: ir.Value) -> ir.Value:
    """abs(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("ReduceL2", self, [-1], keepdims=False)[0]
