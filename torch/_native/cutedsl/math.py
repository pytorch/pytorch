from __future__ import annotations

from typing import TYPE_CHECKING

import cutlass
from cutlass import dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


if TYPE_CHECKING:
    from .types import Scalar


@dsl_user_op
def round_float32(
    x: Scalar, *, loc: ir.Location | None = None, ip: ir.InsertionPoint | None = None
) -> cutlass.Float32:
    """Keep FP32 rounding before a subsequent narrowing conversion."""
    value = cutlass.Float32(x).ir_value(loc=loc, ip=ip)
    return cutlass.Float32(
        llvm.inline_asm(
            value.type,
            [value],
            "",
            "=f,0",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def hypot(
    x: Scalar,
    y: Scalar,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> Scalar:
    r"""Compute CUDA hypot in the operands' FP32 or FP64 type."""
    dtype = type(x)
    if dtype not in (cutlass.Float32, cutlass.Float64) or type(y) is not dtype:
        raise TypeError("hypot requires matching Float32 or Float64 operands")
    symbol = "__nv_hypotf" if dtype is cutlass.Float32 else "__nv_hypot"
    scope = ir.InsertionPoint.current.block.owner
    while scope.operation.name not in ("gpu.module", "builtin.module"):
        scope = scope.parent
    if symbol not in ir.SymbolTable(scope):
        value_type = str(x.ir_value().type)
        signature = ir.Type.parse(
            f"!llvm.func<{value_type} ({value_type}, {value_type})>"
        )
        with ir.InsertionPoint(scope.regions[0].blocks[0]):
            llvm.LLVMFuncOp(symbol, ir.TypeAttr.get(signature))
    value = llvm.call(
        x.ir_value().type,
        [x.ir_value(), y.ir_value()],
        [],
        [],
        callee=symbol,
        loc=loc,
        ip=ip,
    )
    return dtype(value)
