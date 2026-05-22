"""Inline-PTX helpers shared across the TMA and vec-scatter kernels.

Each helper wraps ``llvm.inline_asm`` with the options we always use
(AT&T dialect, side-effecting) and normalizes argument types (cute's
``Numeric`` wrappers get converted to ``ir.Value`` automatically). This
keeps the kernel files focused on algorithm instead of MLIR plumbing.
"""

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T


def as_ir(v, *, loc=None, ip=None):
    """Normalize ``v`` to ``ir.Value``: cute Numeric wrappers (Int32,
    Float16, etc.) get unwrapped via ``.ir_value()``; other values pass
    through unchanged (they're already ir.Values)."""
    if hasattr(v, "ir_value"):
        return v.ir_value(loc=loc, ip=ip)
    return v


def _inline_asm(result_type, operands, asm, constraint, *, loc=None, ip=None):
    """``llvm.inline_asm`` with the flags we always use."""
    return llvm.inline_asm(
        result_type,
        operands,
        asm,
        constraint,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cvta_smem(ptr, *, loc=None, ip=None):
    """Shared-memory ptr -> its ``.shared::cta`` state-space u64 address."""
    raw = ptr.toint(loc=loc, ip=ip).ir_value()
    # cvta is pure, not side-effecting; use inline_asm directly.
    return llvm.inline_asm(
        T.i64(),
        [raw],
        "cvta.to.shared::cta.u64 $0, $1;",
        "=l,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def make_bulk_reduce_add(ptx_suffix: str):
    """Emit ``cp.reduce.async.bulk.global.shared::cta.bulk_group.add.<suffix>``.
    Suffix selects the dtype: ``f32`` / ``noftz.f16`` / ``noftz.bf16``."""

    @dsl_user_op
    def _op(gmem_addr_u64, smem_u64, size_bytes, *, loc=None, ip=None):
        _inline_asm(
            None,
            [
                as_ir(gmem_addr_u64, loc=loc, ip=ip),
                as_ir(smem_u64, loc=loc, ip=ip),
                as_ir(size_bytes, loc=loc, ip=ip),
            ],
            f"cp.reduce.async.bulk.global.shared::cta.bulk_group.add.{ptx_suffix}"
            " [$0], [$1], $2;",
            "l,l,r",
            loc=loc,
            ip=ip,
        )

    return _op


def make_packed_half_atomic_add(ptx_suffix: str):
    """Emit ``red.global.add.noftz.<suffix>`` with a single ``i32`` packed
    register holding two halves. Used by the vec-scatter kernel for
    ``f16x2`` / ``bf16x2`` paired atomics."""

    @dsl_user_op
    def _op(gmem_ptr_i64, packed_i32, *, loc=None, ip=None):
        _inline_asm(
            None,
            [as_ir(gmem_ptr_i64, loc=loc, ip=ip), as_ir(packed_i32, loc=loc, ip=ip)],
            f"red.global.add.noftz.{ptx_suffix} [$0], $1;",
            "l,r",
            loc=loc,
            ip=ip,
        )

    return _op
