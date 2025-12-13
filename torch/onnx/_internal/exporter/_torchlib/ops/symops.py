"""Implementation for torch.sym* ops."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TCH001,TCH002

from __future__ import annotations

from onnxscript.onnx_opset import opset18 as op

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import (
    BOOL,
    FLOAT,
    IntType,
    TensorType,
)
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


@onnx_impl(torch.sym_float, trace_only=True)
def sym_float(self: TensorType) -> FLOAT:
    """sym_float(SymInt self) -> SymFloat"""
    return op.Cast(self, to=FLOAT.dtype)


@onnx_impl(torch.sym_max, trace_only=True)
def sym_max(x: IntType, y: IntType) -> IntType:
    """sym_max(SymInt x, SymInt y) -> SymInt"""
    return op.Max(x, y)


@onnx_impl(torch.sym_min, trace_only=True)
def sym_min(x: IntType, y: IntType) -> IntType:
    """sym_min(SymInt x, SymInt y) -> SymInt"""
    return op.Min(x, y)


@onnx_impl(torch.sym_not, trace_only=True)
def sym_not(self: BOOL) -> BOOL:
    """sym_not(SymBool self) -> SymBool"""
    return op.Not(self)
