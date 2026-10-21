"""Temporary wrappers for operations not yet exposed by FlyDSL."""

from enum import Enum

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith
from flydsl.expr.typing import Float, Integer, Numeric, Pointer


class AtomicOrdering(Enum):
    MONOTONIC = llvm.AtomicOrdering.monotonic
    ACQUIRE = llvm.AtomicOrdering.acquire
    RELEASE = llvm.AtomicOrdering.release
    ACQ_REL = llvm.AtomicOrdering.acq_rel


class AtomicSyncScope(Enum):
    SYSTEM = fx.SyncScope.System
    AGENT = fx.rocdl.SyncScope.Agent
    WORKGROUP = fx.rocdl.SyncScope.Workgroup


__all__ = ["AtomicOrdering", "AtomicSyncScope", "atomic_add", "maxsi", "minsi"]


def maxsi(lhs: Integer, rhs: Integer) -> Integer:
    """Return the signed maximum of two FlyDSL integers."""
    return lhs.dtype(arith.maxsi(arith.unwrap(lhs), arith.unwrap(rhs)))


def minsi(lhs: Integer, rhs: Integer) -> Integer:
    """Return the signed minimum of two FlyDSL integers."""
    return lhs.dtype(arith.minsi(arith.unwrap(lhs), arith.unwrap(rhs)))


def atomic_add(
    ptr: Pointer,
    val: Numeric | int | float,
    *,
    ordering: AtomicOrdering = AtomicOrdering.MONOTONIC,
    syncscope: AtomicSyncScope = AtomicSyncScope.WORKGROUP,
) -> Numeric:
    """Atomically add a scalar value and return the previous value.

    Args:
        ptr: Pointer to the target memory location.
        val: Value to add.
        ordering: Memory ordering semantics.
        syncscope: Threads that participate in the atomic ordering.
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
        ordering.value,
        syncscope=syncscope.value,
        alignment=ptr.alignment,
    ).result
    return dtype(old)
