# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Minimal FlyDSL device helpers required by the vendored RMSNorm kernels.

These helpers are copied from ``ROCm/FlyDSL:kernels/common/kernels_common.py``
instead of importing its example ``kernels`` package.  A normal FlyDSL wheel
installs the compiler/runtime package, but does not make a separate FlyDSL
source checkout a runtime dependency of PyTorch.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith as _expr_arith
from flydsl.expr import const_expr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch


def _get_llvm_ptr(ptr, offset, dtype_bytes, ptr_type=None):
    """Return a global-memory LLVM pointer at ``ptr + offset*dtype_bytes``."""

    if ptr_type is None:
        ptr_type = ir.Type.parse("!llvm.ptr<1>")
    base_ptr = _fly.extract_aligned_pointer_as_index(ptr_type, ptr)
    base_ptr = _llvm.PtrToIntOp(T.i64, base_ptr).result
    byte_offset = _expr_arith.index_cast(
        T.i64, fx.Index(offset) * fx.Index(dtype_bytes)
    )
    llvm_ptr = _llvm.AddOp(
        base_ptr, byte_offset, _llvm.IntegerOverflowFlags(0)
    ).result
    llvm_ptr = _llvm.IntToPtrOp(ptr_type, llvm_ptr).result
    return llvm_ptr._value if const_expr(hasattr(llvm_ptr, "_value")) else llvm_ptr


def atomic_add(
    dst,
    offset,
    value,
    *,
    dtype_bytes=4,
    syncscope="agent",
    ordering=None,
    alignment=None,
    ptr_type=None,
):
    """Atomically add ``value`` to ``dst[offset]`` in global memory."""

    ptr = _get_llvm_ptr(dst, offset, dtype_bytes, ptr_type=ptr_type)
    val = value.ir_value() if const_expr(hasattr(value, "ir_value")) else value
    elem_ty = val.type.element_type if isinstance(val.type, ir.VectorType) else val.type
    bin_op = (
        _llvm.AtomicBinOp.fadd
        if isinstance(elem_ty, ir.FloatType)
        else _llvm.AtomicBinOp.add
    )
    if ordering is None:
        ordering = _llvm.AtomicOrdering.monotonic
    if alignment is None:
        alignment = dtype_bytes
    return _llvm.AtomicRMWOp(
        bin_op,
        ptr,
        val,
        ordering,
        syncscope=syncscope,
        alignment=alignment,
    ).result


def dtype_to_elem_type(dtype_str: str):
    """Map the three supported PyTorch dtype strings to FlyDSL types."""

    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(
        f"unsupported dtype: {dtype_str!r} "
        "(expected 'f32', 'f16', or 'bf16')"
    )


def get_warp_size(arch=None) -> int:
    """Return wave64 for CDNA GPUs and wave32 for RDNA GPUs."""

    if arch is None:
        arch = get_rocm_arch()
    return 32 if is_rdna_arch(arch) else 64