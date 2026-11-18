# Copyright (c) 2026, Tri Dao.
"""Small CuTe tensor convenience helpers.

Importing this module intentionally mutates CuTe's tensor class process-wide so
fragments can be written in a more PyTorch-like style::

    rmem_f32 = rmem_f16.to(Float32)
    rmem_copy = rmem_view.clone()
    rmem_contig = rmem_view.contiguous()

``tensor.to(dtype)`` is exactly the explicit CuTe sequence::

    dst = cute.make_rmem_tensor_like(src, dtype)
    dst.store(src.load().to(dtype))

``tensor.to(dtype, force_materialize=True)`` uses the same value conversion, then inserts an
opaque no-op SSA boundary on packed f16/bf16 lanes.  This is useful when downstream codegen
would otherwise rematerialize a vector truncation for multiple consumers.

``tensor.clone()`` materializes into ``cute.make_rmem_tensor_like(src)``.
``tensor.contiguous()`` mirrors ``quack.copy_utils.contiguous``.
"""

from __future__ import annotations

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.cute.tensor as _cute_tensor
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


_ORIGINAL_TO_ATTR = "_quack_original_to"
_PATCHED_TO_ATTR = "_quack_rmem_tensor_to"
_ORIGINAL_CLONE_ATTR = "_quack_original_clone"
_PATCHED_CLONE_ATTR = "_quack_tensor_clone"
_ORIGINAL_CONTIGUOUS_ATTR = "_quack_original_contiguous"
_PATCHED_CONTIGUOUS_ATTR = "_quack_tensor_contiguous"


@dsl_user_op
def _black_box_b32(x: cutlass.Int32, *, loc: Any = None, ip: Any = None) -> cutlass.Int32:
    """Opaque identity for packed registers.

    The empty asm emits no PTX instruction.  The tied input constraint (``0``) makes the
    output use the same register class/value as the input, while still creating an SSA boundary.
    """

    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int32(x).ir_value(loc=loc, ip=ip)],
            "",
            "=r,0",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _to_f16_materialized(src: Any, dtype: Any) -> Any:
    assert src.element_type is cutlass.Float32, "src must be Float32"
    assert dtype in (cutlass.BFloat16, cutlass.Float16), "dtype must be BFloat16 or Float16"
    assert cute.size(src.shape) % 2 == 0, "src must have an even number of elements"

    # Why this exists: plain Tensor.to lowers f32 -> f16/bf16 as a vector truncation.
    # In some kernels the converted fragment has multiple consumers (e.g. STSM and RS WGMMA
    # in SM90 FlashAttention backward).  Leaving the truncation as a high-level vector value
    # lets later codegen rematerialize it for each consumer, producing extra packed converts
    # and a worse WGMMA schedule.  Storing the .to result, viewing the f16/bf16 pairs as i32,
    # then passing each packed lane through an empty tied-operand asm creates an opaque SSA
    # boundary.  The asm emits no PTX instruction, but it forces one materialized packed value
    # that downstream users share.
    tmp = cute.make_rmem_tensor_like(src, dtype)
    tmp.store(src.load().to(dtype))
    dst = cute.make_rmem_tensor_like(src, dtype)
    tmp_i32 = cute.recast_tensor(tmp, cutlass.Int32)
    dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
    assert cute.size(dst_i32.shape) * 2 == cute.size(src.shape)
    for i in cutlass.range(cute.size(dst_i32), unroll_full=True):
        dst_i32[i] = _black_box_b32(tmp_i32[i])
    return dst


def _make_to() -> Any:
    @dsl_user_op
    def _to(
        self: Any,
        dtype: Any,
        *,
        force_materialize: bool = False,
        loc: Any = None,
        ip: Any = None,
    ) -> Any:
        if self.memspace != cute.AddressSpace.rmem:
            raise ValueError("Tensor.to(dtype) is only supported for rmem tensors")

        if force_materialize:
            return _to_f16_materialized(self, dtype)

        dst = cute.make_rmem_tensor_like(self, dtype, loc=loc, ip=ip)
        dst.store(self.load(loc=loc, ip=ip).to(dtype, loc=loc, ip=ip), loc=loc, ip=ip)
        return dst

    return _to


def _make_clone() -> Any:
    @dsl_user_op
    def _clone(self: Any, *, loc: Any = None, ip: Any = None) -> Any:
        dst = cute.make_rmem_tensor_like(self, loc=loc, ip=ip)
        cute.autovec_copy(self, dst, loc=loc, ip=ip)
        return dst

    return _clone


def _make_contiguous() -> Any:
    @dsl_user_op
    def _contiguous(self: Any, *, loc: Any = None, ip: Any = None) -> Any:
        dst = cute.make_rmem_tensor(self.shape, self.element_type, loc=loc, ip=ip)
        cute.autovec_copy(self, dst, loc=loc, ip=ip)
        return dst

    return _contiguous


def patch_cute_tensor() -> None:
    """Monkey patch CuTe tensors with QuACK convenience methods.

    The patch is idempotent. CuTe's immutable ``TensorSSA.to`` already handles
    value conversion; this installs the analogous materializing conversion on
    mutable register-backed ``_Tensor`` fragments, plus a ``contiguous`` method
    equivalent to :func:`quack.copy_utils.contiguous`, and a ``clone`` method
    that copies into a matching compact rmem tensor.
    """
    tensor_cls = _cute_tensor._Tensor
    if _PATCHED_TO_ATTR not in tensor_cls.__dict__:
        original_to = getattr(tensor_cls, "to", None)
        if original_to is not None:
            setattr(tensor_cls, _ORIGINAL_TO_ATTR, original_to)
        tensor_cls.to = _make_to()  # type: ignore[method-assign]
        setattr(tensor_cls, _PATCHED_TO_ATTR, True)

    if _PATCHED_CLONE_ATTR not in tensor_cls.__dict__:
        original_clone = getattr(tensor_cls, "clone", None)
        if original_clone is not None:
            setattr(tensor_cls, _ORIGINAL_CLONE_ATTR, original_clone)
        tensor_cls.clone = _make_clone()  # type: ignore[method-assign]
        setattr(tensor_cls, _PATCHED_CLONE_ATTR, True)

    if _PATCHED_CONTIGUOUS_ATTR not in tensor_cls.__dict__:
        original_contiguous = getattr(tensor_cls, "contiguous", None)
        if original_contiguous is not None:
            setattr(tensor_cls, _ORIGINAL_CONTIGUOUS_ATTR, original_contiguous)
        tensor_cls.contiguous = _make_contiguous()  # type: ignore[method-assign]
        setattr(tensor_cls, _PATCHED_CONTIGUOUS_ATTR, True)


patch_cute_tensor()


__all__ = ["patch_cute_tensor"]
