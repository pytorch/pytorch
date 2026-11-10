"""Implementation for torch.sym* ops."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TC003

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


if TYPE_CHECKING:
    from torch.onnx._internal.exporter._ir_builder import TranslationContext


@onnx_impl(torch.sym_float)
def sym_float(op: TranslationContext, self: ir.Value) -> ir.Value:
    """sym_float(SymInt self) -> SymFloat"""
    return op.Cast(self, to=ir.DataType.FLOAT)


@onnx_impl(torch.sym_max)
def sym_max(op: TranslationContext, x: ir.Value, y: ir.Value) -> ir.Value:
    """sym_max(SymInt x, SymInt y) -> SymInt"""
    return op.Max(x, y)


@onnx_impl(torch.sym_min)
def sym_min(op: TranslationContext, x: ir.Value, y: ir.Value) -> ir.Value:
    """sym_min(SymInt x, SymInt y) -> SymInt"""
    return op.Min(x, y)


@onnx_impl(torch.sym_not)
def sym_not(op: TranslationContext, self: ir.Value) -> ir.Value:
    """sym_not(SymBool self) -> SymBool"""
    return op.Not(self)


@onnx_impl(torch.sym_sum)
def sym_sum(op: TranslationContext, args: Sequence[ir.Value]) -> ir.Value:
    """sym_sum(SymInt[] args) -> SymInt"""
    if len(args) == 0:
        return op.Constant(value_int=0)
    if len(args) == 1:
        return args[0]
    result = op.Add(args[0], args[1])
    for i in range(2, len(args)):
        result = op.Add(result, args[i])
    return result


@onnx_impl(torch.sym_ite)
def sym_ite(op: TranslationContext, b: ir.Value, t: ir.Value, f: ir.Value) -> ir.Value:
    """sym_ite(SymBool b, Tensor t, Tensor f) -> Tensor"""
    return op.Where(b, t, f)
