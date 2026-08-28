"""Temporary wrappers for operations not yet exposed by FlyDSL."""

from typing import Literal

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith
from flydsl.expr.typing import Float, Integer, Numeric, Pointer


AtomicSem = Literal["relaxed", "release", "acquire", "acq_rel"]
AtomicScope = Literal["gpu", "cta", "sys"]

__all__ = ["AtomicScope", "AtomicSem", "atomic_add"]

_ATOMIC_ORDERINGS = {
    "relaxed": llvm.AtomicOrdering.monotonic,
    "release": llvm.AtomicOrdering.release,
    "acquire": llvm.AtomicOrdering.acquire,
    "acq_rel": llvm.AtomicOrdering.acq_rel,
}
_ATOMIC_SCOPES = {
    "gpu": fx.rocdl.SyncScope.Agent,
    "cta": fx.rocdl.SyncScope.Workgroup,
    "sys": fx.SyncScope.System,
}


def atomic_add(
    ptr: Pointer,
    val: Numeric | int | float,
    *,
    sem: AtomicSem = "relaxed",
    scope: AtomicScope = "cta",
) -> Numeric:
    """Atomically add a scalar value and return the previous value.

    Args:
        ptr: Pointer to the target memory location.
        val: Value to add.
        sem: Memory ordering semantics.
        scope: Threads that participate in the atomic ordering.
    """
    dtype = ptr.element_type
    if not issubclass(dtype, (Integer, Float)) or dtype.width < 8:
        raise TypeError(f"atomic_add does not support {dtype}")
    typed_val = dtype(val)
    op = llvm.AtomicBinOp.fadd if issubclass(dtype, Float) else llvm.AtomicBinOp.add
    old = llvm.AtomicRMWOp(
        op,
        fx.to_llvm_ptr(ptr),
        arith.unwrap(typed_val),
        _ATOMIC_ORDERINGS[sem],
        syncscope=_ATOMIC_SCOPES[scope],
        alignment=dtype.width // 8,
    ).result
    return dtype(old)
