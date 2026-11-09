"""Implementation for torch.sym* ops."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TC003

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.onnx._internal.exporter._ir_builder import current_builder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


@onnx_impl(torch.sym_float)
def sym_float(self: ir.Value) -> ir.Value:
    """sym_float(SymInt self) -> SymFloat"""
    builder = current_builder()
    return builder.call("Cast", self, to=ir.DataType.FLOAT)[0]


@onnx_impl(torch.sym_max)
def sym_max(x: ir.Value, y: ir.Value) -> ir.Value:
    """sym_max(SymInt x, SymInt y) -> SymInt"""
    builder = current_builder()
    return builder.call("Max", x, y)[0]


@onnx_impl(torch.sym_min)
def sym_min(x: ir.Value, y: ir.Value) -> ir.Value:
    """sym_min(SymInt x, SymInt y) -> SymInt"""
    builder = current_builder()
    return builder.call("Min", x, y)[0]


@onnx_impl(torch.sym_not)
def sym_not(self: ir.Value) -> ir.Value:
    """sym_not(SymBool self) -> SymBool"""
    builder = current_builder()
    return builder.call("Not", self)[0]


@onnx_impl(torch.sym_sum)
def sym_sum(args: Sequence[ir.Value]) -> ir.Value:
    """sym_sum(SymInt[] args) -> SymInt"""
    builder = current_builder()
    if len(args) == 0:
        return builder.call("Constant", value_int=0)[0]
    if len(args) == 1:
        return args[0]
    result = builder.call("Add", args[0], args[1])[0]
    for i in range(2, len(args)):
        result = builder.call("Add", result, args[i])[0]
    return result


@onnx_impl(torch.sym_ite)
def sym_ite(b: ir.Value, t: ir.Value, f: ir.Value) -> ir.Value:
    """sym_ite(SymBool b, Tensor t, Tensor f) -> Tensor"""
    builder = current_builder()
    return builder.call("Where", b, t, f)[0]
