# Copyright (c) 2025, Tri Dao.
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Boolean, const_expr
from cutlass.cute.nvgpu import tcgen05
from cutlass._mlir.dialects import llvm

from . import mma_sm100_desc as sm100_desc


def _tcgen05_mma_kind(op: cute.nvgpu.tcgen05.mma.MmaOp) -> str:
    if isinstance(op, tcgen05.mma.MmaF16BF16Op):
        return "f16"
    if isinstance(op, tcgen05.mma.MmaTF32Op):
        return "tf32"
    if isinstance(op, tcgen05.mma.MmaI8Op):
        return "i8"
    # cutlass-dsl >=4.5.2 builds plain FP8 MMAs as MmaF8F6F4Op (make_trivial_tiled_mma's
    # _F8F6F4_TYPES branch); <4.4.x returned the now-legacy MmaFP8Op. Both map to kind::f8f6f4.
    if isinstance(op, (tcgen05.mma.MmaFP8Op, tcgen05.mma.MmaF8F6F4Op)):
        return "f8f6f4"
    if isinstance(op, tcgen05.mma.MmaMXF8Op):
        return "mxf8f6f4"
    if isinstance(op, tcgen05.mma.MmaMXF4Op):
        return "mxf4"
    if isinstance(op, tcgen05.mma.MmaMXF4NVF4Op):
        return "mxf4nvf4"
    raise TypeError(f"Unsupported tcgen05 MMA op kind: {type(op).__name__}")


@cute.jit
def gemm_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    zero_init: bool | Boolean = False,
    swap_AB: bool = False,
    num_unroll_groups: int = 1,
) -> None:
    if const_expr(swap_AB):
        return gemm_w_idx(
            tiled_mma, acc, tCrB, tCrA, B_idx, A_idx, zero_init=zero_init, swap_AB=False
        )
    else:
        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]

        mma_atom = cute.make_mma_atom(tiled_mma.op)
        for k in cutlass.range(
            cute.size(tCrA.shape[2]), unroll=cute.size(tCrA.shape[2]) // num_unroll_groups
        ):
            mma_atom.set(tcgen05.Field.ACCUMULATE, not zero_init or k != 0)
            cute.gemm(mma_atom, acc, rA[None, None, k], rB[None, None, k], acc)


@cute.jit
def gemm_ptx_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    zero_init: bool | Boolean = False,
    cta_group: int = 1,
    **kwargs,
) -> None:
    rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    sA_cur = None
    if const_expr(sA is not None):
        sA_cur = sA if const_expr(A_idx is None) else sA[None, None, None, A_idx]
    sB_cur = sB if const_expr(B_idx is None) else sB[None, None, None, B_idx]
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    acc_tmem_addr = acc.iterator.toint()
    gemm_ptx_partial(
        mma_atom.op,
        acc_tmem_addr,
        rA,
        rB,
        sA_cur,
        sB_cur,
        zero_init=zero_init,
        cta_group=cta_group,
        **kwargs,
    )


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: bool | Boolean = False,
) -> None:
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        mma_atom.set(tcgen05.Field.ACCUMULATE, not zero_init or k != 0)
        cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


def i64_to_i32x2(i: int) -> Tuple[int, int]:
    """Convert a 64-bit integer to a tuple of two 32-bit integers."""
    return i & 0xFFFF_FFFF, (i >> 32) & 0xFFFF_FFFF


@cute.jit
def gemm_ptx(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    zero_init: bool | Boolean = False,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else None
    sB_layout = sB.layout
    idesc: int = const_expr(sm100_desc.mma_op_to_idesc(op))
    kind = _tcgen05_mma_kind(op)
    if const_expr(not is_ts):
        sA_swizzle = sA.iterator.type.swizzle_type
        smem_desc_base_a: int = const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                sm100_desc.Major.K
                if const_expr(op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
                else sm100_desc.Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    sB_swizzle = sB.iterator.type.swizzle_type
    smem_desc_base_b: int = const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            sm100_desc.Major.K
            if const_expr(op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
            else sm100_desc.Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)

    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(smem_desc_base_a_lo) | sm100_desc.make_smem_desc_start_addr(
            sA[None, None, 0].iterator
        )
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = Int32(smem_desc_base_b_lo) | sm100_desc.make_smem_desc_start_addr(
        sB[None, None, 0].iterator
    )
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if const_expr(not is_ts):
            smem_desc_a_lo = smem_desc_start_a_lo + (
                (cute.crd2idx((0, 0, k), sA_layout) * sA.element_type.width // 8) >> 4
            )
        smem_desc_b_lo = smem_desc_start_b_lo + (
            (cute.crd2idx((0, 0, k), sB_layout) * sB.element_type.width // 8) >> 4
        )
        # with cute.arch.elect_one():
        #     cute.printf("smem_desc_a_lo = {}, smem_desc_b_lo = {}", smem_desc_a_lo, smem_desc_b_lo)
        #     cute.printf("smem_desc_a_lo_correct = {}, smem_desc_b_lo_correct = {}", smem_desc_a_lo_correct, smem_desc_b_lo_correct)
        with cute.arch.elect_one():
            if const_expr(not is_ts):
                llvm.inline_asm(
                    None,
                    [
                        acc.iterator.toint().ir_value(),
                        smem_desc_a_lo.ir_value(),
                        smem_desc_b_lo.ir_value(),
                        Int32(not zero_init or k != 0).ir_value(),
                    ],
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
                    ".reg .b32 idesc;\n\t"
                    f"mov.b32 idesc, {hex(idesc)};\n\t"
                    f"mov.b64 smem_desc_a, {{$1, {hex(smem_desc_a_hi)}}};\n\t"
                    f"mov.b64 smem_desc_b, {{$2, {hex(smem_desc_b_hi)}}};\n\t"
                    "setp.ne.b32 p, $3, 0;\n\t"
                    f"tcgen05.mma.cta_group::1.kind::{kind} [$0], smem_desc_a, smem_desc_b, idesc, p;\n\t"
                    "}\n",
                    "r,r,r,r",
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            else:
                llvm.inline_asm(
                    None,
                    [
                        acc.iterator.toint().ir_value(),
                        tCrA[None, None, k].iterator.toint().ir_value(),
                        smem_desc_b_lo.ir_value(),
                        Int32(not zero_init or k != 0).ir_value(),
                    ],
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    ".reg .b64 smem_desc_b;\n\t"
                    f"mov.b64 smem_desc_b, {{$2, {hex(smem_desc_b_hi)}}};\n\t"
                    "setp.ne.b32 p, $3, 0;\n\t"
                    f"tcgen05.mma.cta_group::1.kind::{kind} [$0], [$1], smem_desc_b, {hex(idesc)}, p;\n\t"
                    "}\n",
                    "r,r,r,r",
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )


@cute.jit
def gemm_ptx_loop(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    zero_init: bool | Boolean = False,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = const_expr(sm100_desc.mma_op_to_idesc(op))
    kind = _tcgen05_mma_kind(op)
    if const_expr(not is_ts):
        sA_swizzle = sA.iterator.type.swizzle_type
        smem_desc_base_a: int = const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                sm100_desc.Major.K
                if const_expr(op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
                else sm100_desc.Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    sB_swizzle = sB.iterator.type.swizzle_type
    smem_desc_base_b: int = const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            sm100_desc.Major.K
            if const_expr(op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
            else sm100_desc.Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)

    if const_expr(not is_ts):
        offset_a = [
            (cute.crd2idx((0, 0, k), sA_layout) * sA.element_type.width // 8) >> 4
            for k in cutlass.range_constexpr(cute.size(tCrA.shape[2]))
        ]
    else:
        offset_a = [
            cute.crd2idx((0, 0, k), sA_layout) * op.a_dtype.width // 32
            for k in cutlass.range_constexpr(cute.size(tCrA.shape[2]))
        ]
    offset_a_diff = [
        offset_a[k] - offset_a[k - 1] for k in cutlass.range_constexpr(1, cute.size(tCrA.shape[2]))
    ]
    offset_b = [
        (cute.crd2idx((0, 0, k), sB_layout) * sB.element_type.width // 8) >> 4
        for k in cutlass.range_constexpr(cute.size(tCrB.shape[2]))
    ]
    offset_b_diff = [
        offset_b[k] - offset_b[k - 1] for k in cutlass.range_constexpr(1, cute.size(tCrB.shape[2]))
    ]

    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(
            smem_desc_base_a_lo | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator)
        )
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = Int32(
        smem_desc_base_b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator)
    )
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [
                acc.iterator.toint().ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(not zero_init).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            "mov.b32 smem_desc_a_lo, $1;\n\t"
            "mov.b32 smem_desc_b_lo, $2;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $3, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [$0], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [$0], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in cutlass.range_constexpr(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        llvm.inline_asm(
            None,
            [
                acc.iterator.toint().ir_value(),
                Int32(tCrA[None, None, 0].iterator.toint()).ir_value(),
                Int32(smem_desc_start_b_lo).ir_value(),
                Int32(not zero_init).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            "mov.b32 tmem_a, $1;\n\t"
            "mov.b32 smem_desc_b_lo, $2;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $3, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [$0], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    # f"add.u32 tmem_a, tmem_a, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    # f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [$0], [tmem_a], smem_desc_b, idesc, 1;\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [$0], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in cutlass.range_constexpr(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def gemm_ptx_partial(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: Int32,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    mbar_ptr: Optional[cutlass.Pointer] = None,
    mbar_phase: Optional[Int32] = None,
    split_arrive: Optional[int] = None,
    zero_init: bool | Boolean = False,
    # sA_offset: Int32 = 0,
    # acc_offset: Int32 = 0,
    tA_addr: Optional[Int32] = None,
    cta_group: int = 1,
) -> None:
    # acc_tmem_addr += acc_offset
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = const_expr(sm100_desc.mma_op_to_idesc(op))
    kind = _tcgen05_mma_kind(op)
    if const_expr(not is_ts):
        sA_swizzle = sA.iterator.type.swizzle_type
        smem_desc_base_a: int = const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                sm100_desc.Major.K
                if const_expr(op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
                else sm100_desc.Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    sB_swizzle = sB.iterator.type.swizzle_type
    smem_desc_base_b: int = const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            sm100_desc.Major.K
            if const_expr(op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
            else sm100_desc.Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)

    tCrA_layout = (
        tCrA.layout
        if const_expr(not is_ts)
        else cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
    )
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(cute.size(tCrA.shape[2]))]
    offset_a_diff = [offset_a[k] - offset_a[k - 1] for k in range(1, cute.size(tCrA.shape[2]))]
    offset_b = [cute.crd2idx((0, 0, k), tCrB.layout) for k in range(cute.size(tCrB.shape[2]))]
    offset_b_diff = [offset_b[k] - offset_b[k - 1] for k in range(1, cute.size(tCrB.shape[2]))]

    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(
            smem_desc_base_a_lo | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator)
        )
        # ) + sA_offset
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = Int32(
        smem_desc_base_b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator)
    )
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                # acc.iterator.toint().ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(not zero_init).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo_start, smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            # f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 smem_desc_a_lo_start, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo_start, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    # f"add.u32 smem_desc_a_lo, smem_desc_a_lo, {hex(offset_a_diff[k - 1])};\n\t"
                    # f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo_start, {hex(offset_a[k])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            # "r,r,r",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        # For TS gemm, somehow tCrA.iterator.toint() returns 0 no matter what, so we need to
        # explicitly pass in the tA_addr for correctness.
        tA_addr = tCrA[None, None, 0].iterator.toint() if tA_addr is None else tA_addr
        input_args = [
            # Int32(cute.arch.make_warp_uniform(tCrA[None, None, 0].iterator.toint())).ir_value(),
            Int32(cute.arch.make_warp_uniform(tA_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(not zero_init).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ]
        if const_expr(mbar_ptr is not None):
            assert mbar_phase is not None, "mbar_phase must be provided when mbar_ptr is not None"
            assert split_arrive is not None, (
                "split_arrive must be provided when mbar_ptr is not None"
            )
            split_arrive_idx = split_arrive // op.shape_mnk[2]
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [$4], $5, 10000000; \n\t"
                "@P1 bra DONE; \n\t"
                "bra     LAB_WAIT; \n\t"
                "DONE: \n\t"
            )
        else:
            mbar_wait_str = ""
        llvm.inline_asm(
            None,
            # [
            #     # acc.iterator.toint().ir_value(),
            #     Int32(tCrA[None, None, 0].iterator.toint()).ir_value(),
            #     Int32(smem_desc_start_b_lo).ir_value(),
            #     Int32(not zero_init).ir_value(),
            # ],
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            # f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            f"mov.b32 tmem_a, $0;\n\t"
            f"mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    # f"add.u32 tmem_a, tmem_a, {hex(offset_a_diff[k - 1])};\n\t"
                    # f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    # f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a], smem_desc_b, idesc, 1;\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(
                    1,
                    cute.size(tCrA.shape[2]) if const_expr(mbar_ptr is None) else split_arrive_idx,
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                        f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                    )
                    for k in range(split_arrive_idx, cute.size(tCrA.shape[2]))
                )
                if const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\n",
            "r,r,r,r" if const_expr(mbar_ptr is None) else "r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def gemm_ptx_partial1(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: cutlass.Constexpr[int],
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA_base_addr_for_desc: Int32,
    sA_addr_offset_for_desc: cutlass.Constexpr[int],
    sA_stage: Int32,
    sB_base_addr_for_desc: Int32,
    sB_addr_offset_for_desc: cutlass.Constexpr[int],
    sB_stage: Int32,
    sA_layout: Optional[cute.Layout],
    sB_layout: Optional[cute.Layout],
    sA_swizzle: Optional[cute.Swizzle],
    sB_swizzle: cute.Swizzle,
    zero_init: bool | Boolean = False,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA_layout is not None, "sA_layout must be provided when a_src is not TMEM"
        assert sA_swizzle is not None, "sA_swizzle must be provided when a_src is not TMEM"
    idesc: int = const_expr(sm100_desc.mma_op_to_idesc(op))
    kind = _tcgen05_mma_kind(op)
    if const_expr(not is_ts):
        smem_desc_base_a: int = const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                sm100_desc.Major.K
                if const_expr(op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
                else sm100_desc.Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    smem_desc_base_b: int = const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            sm100_desc.Major.K
            if const_expr(op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
            else sm100_desc.Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)
    mask = [Int32(0)] * 4

    if const_expr(not is_ts):
        offset_a = [
            (cute.crd2idx((0, 0, k), sA_layout) * op.a_dtype.width // 8) >> 4
            for k in range(cute.size(tCrA.shape[2]))
        ]
    else:
        offset_a = [
            cute.crd2idx((0, 0, k), sA_layout) * op.a_dtype.width // 32
            for k in range(cute.size(tCrA.shape[2]))
        ]
    offset_a_diff = [offset_a[k] - offset_a[k - 1] for k in range(1, cute.size(tCrA.shape[2]))]
    offset_b = [
        (cute.crd2idx((0, 0, k), sB_layout) * op.b_dtype.width // 8) >> 4
        for k in range(cute.size(tCrB.shape[2]))
    ]
    offset_b_diff = [offset_b[k] - offset_b[k - 1] for k in range(1, cute.size(tCrB.shape[2]))]

    if const_expr(not is_ts):
        # smem_desc_start_a_lo = Int32(smem_desc_base_a_lo | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator))
        smem_desc_start_a_lo = const_expr(smem_desc_base_a_lo)
    else:
        smem_desc_start_a_lo = None
    # smem_desc_start_b_lo = Int32(smem_desc_base_b_lo | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator))
    smem_desc_start_b_lo = const_expr(smem_desc_base_b_lo)
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [
                # acc.iterator.toint().ir_value(),
                # Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(sA_base_addr_for_desc).ir_value(),
                Int32(sA_stage).ir_value(),
                # Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(sB_base_addr_for_desc).ir_value(),
                Int32(sB_stage).ir_value(),
                Int32(not zero_init).ir_value(),
                mask[0].ir_value(),
                mask[1].ir_value(),
                mask[2].ir_value(),
                mask[3].ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            # "mov.b32 smem_desc_a_lo, $0;\n\t"
            # f"add.u32 smem_desc_a_lo, $0, {hex(smem_desc_start_a_lo)};\n\t"
            f"mad.lo.u32 smem_desc_a_lo, $1, {hex(sA_addr_offset_for_desc)}, $0;\n\t"
            # "mov.b32 smem_desc_b_lo, $2;\n\t"
            f"mad.lo.u32 smem_desc_b_lo, $3, {hex(sB_addr_offset_for_desc)}, $2;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $4, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, {{$5, $6, $7, $8}}, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, {{$5, $6, $7, $8}}, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        llvm.inline_asm(
            None,
            [
                # acc.iterator.toint().ir_value(),
                Int32(tCrA[None, None, 0].iterator.toint()).ir_value(),
                Int32(smem_desc_start_b_lo).ir_value(),
                Int32(not zero_init).ir_value(),
                mask[0].ir_value(),
                mask[1].ir_value(),
                mask[2].ir_value(),
                mask[3].ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_a, $1;\n\t"
            f"mov.b32 smem_desc_b_lo, $2;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $3, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [$0], [tmem_a], smem_desc_b, idesc, {{$4, $5, $6, $7}}, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 tmem_a, tmem_a, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [$0], [tmem_a], smem_desc_b, idesc, {{$4, $5, $6, $7}}, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def gemm_ptx_precomputed(
    acc_tmem_addr: Int32,
    smem_desc_start_a: Int32,  # If TS, then this is the tmem start address for A
    smem_desc_start_b: Int32,
    idesc: int,
    smem_desc_base_a: Optional[int],
    smem_desc_base_b: int,
    tCrA_layout: cute.Layout,
    tCrB_layout: cute.Layout,
    mbar_ptr: Optional[cutlass.Pointer] = None,
    mbar_phase: Optional[Int32] = None,
    zero_init: bool | Boolean = False,
    cta_group: int = 1,
    kind: str = "f16",
) -> None:
    # acc_tmem_addr += acc_offset
    is_ts = const_expr(smem_desc_base_a is None)
    num_k_tile = cute.size(tCrA_layout.shape[2])
    if const_expr(not is_ts):
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
    else:
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)

    tCrA_layout = (
        tCrA_layout
        if const_expr(not is_ts)
        # else cute.recast_layout(32, tCrA.element_type.width, tCrA_layout)
        # currently hard-coding the width to 16
        else cute.recast_layout(32, 16, tCrA_layout)
    )
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(num_k_tile)]
    offset_a_diff = [offset_a[k] - offset_a[k - 1] for k in range(1, num_k_tile)]
    offset_b = [cute.crd2idx((0, 0, k), tCrB_layout) for k in range(num_k_tile)]
    offset_b_diff = [offset_b[k] - offset_b[k - 1] for k in range(1, num_k_tile)]

    smem_desc_start_a_lo = None
    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(smem_desc_base_a_lo | smem_desc_start_a)
        # smem_desc_start_a_lo = smem_desc_start_a
    smem_desc_start_b_lo = Int32(smem_desc_base_b_lo | smem_desc_start_b)
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                # acc.iterator.toint().ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(not zero_init).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo_start, smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            # f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 smem_desc_a_lo_start, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo_start, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    # f"add.u32 smem_desc_a_lo, smem_desc_a_lo, {hex(offset_a_diff[k - 1])};\n\t"
                    # f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"add.s32 smem_desc_a_lo, smem_desc_a_lo_start, {hex(offset_a[k])};\n\t"
                    f"add.s32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(1, num_k_tile)
            )
            + "}\n",
            # "r,r,r",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        input_args = [
            Int32(cute.arch.make_warp_uniform(smem_desc_start_a)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(not zero_init).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ]
        if const_expr(mbar_ptr is not None):
            assert mbar_phase is not None, "mbar_phase must be provided when mbar_ptr is not None"
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [$4], $5, 10000000; \n\t"
                "@P1 bra DONE; \n\t"
                "bra     LAB_WAIT; \n\t"
                "DONE: \n\t"
            )
        else:
            mbar_wait_str = ""
        llvm.inline_asm(
            None,
            # [
            #     # acc.iterator.toint().ir_value(),
            #     Int32(tCrA_layout[None, None, 0].iterator.toint()).ir_value(),
            #     Int32(smem_desc_start_b_lo).ir_value(),
            #     Int32(not zero_init).ir_value(),
            # ],
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            # f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            f"mov.b32 tmem_a, $0;\n\t"
            f"mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    # f"add.u32 tmem_a, tmem_a, {hex(offset_a_diff[k - 1])};\n\t"
                    # f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    # f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a], smem_desc_b, idesc, 1;\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(
                    1,
                    num_k_tile if const_expr(mbar_ptr is None) else num_k_tile // 4 * 3,
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        # f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                        f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                    )
                    for k in range(num_k_tile // 4 * 3, num_k_tile)
                )
                if const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\n",
            "r,r,r,r" if const_expr(mbar_ptr is None) else "r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def declare_ptx_smem_desc(
    smem_desc_start_a: Int32,  # If TS, then this is the tmem start address for A
    smem_desc_base_a: Optional[int],
    tCrA_layout: cute.Layout,
    var_name_prefix: str = "smem_desc",
) -> None:
    is_ts = const_expr(smem_desc_base_a is None)
    num_k_tile = cute.size(tCrA_layout.shape[2])
    smem_desc_base_a_lo, smem_desc_a_hi = None, None
    if const_expr(not is_ts):
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
    tCrA_layout = (
        tCrA_layout
        if const_expr(not is_ts)
        # else cute.recast_layout(32, tCrA.element_type.width, tCrA_layout)
        # currently hard-coding the width to 16
        else cute.recast_layout(32, 16, tCrA_layout)
    )
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(num_k_tile)]
    smem_desc_start_a_lo = None
    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(smem_desc_base_a_lo | smem_desc_start_a)
    if const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value()],
            f".reg .b32 {var_name_prefix}_lo;\n\t"
            f".reg .b64 {var_name_prefix}_<{num_k_tile}>;\n\t"
            f"mov.b64 {var_name_prefix}_0, {{$0, {hex(smem_desc_a_hi)}}};\n\t"
            + "".join(
                (
                    f"add.s32 {var_name_prefix}_lo, $0, {hex(offset_a[k])};\n\t"
                    f"mov.b64 {var_name_prefix}_{k}, {{{var_name_prefix}_lo, {hex(smem_desc_a_hi)}}};\n\t"
                )
                for k in range(1, num_k_tile)
            ),
            "r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def declare_ptx_idesc(op: cute.nvgpu.tcgen05.mma.MmaOp, var_name: str = "idesc") -> None:
    idesc = const_expr(sm100_desc.mma_op_to_idesc(op))
    llvm.inline_asm(
        None,
        [],
        f".reg .b32 {var_name};\n\t"  # noqa
        f"mov.b32 {var_name}, {hex(idesc)};\n\t",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def gemm_ptx_precomputed_varname(
    acc_tmem_addr: Int32,
    smem_desc_start_b: Int32,
    # idesc: int,
    smem_desc_base_b: int,
    tCrB_layout: cute.Layout,
    smem_var_name_prefix: str,
    idesc_var_name: str,
    smem_offset: int,
    zero_init: bool | Boolean = False,
    cta_group: int = 1,
    kind: str = "f16",
) -> None:
    is_ts = False
    num_k_tile = cute.size(tCrB_layout.shape[2])
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    offset_b = [cute.crd2idx((0, 0, k), tCrB_layout) for k in range(num_k_tile)]

    smem_desc_start_b_lo = Int32(smem_desc_base_b_lo | smem_desc_start_b)
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(not zero_init).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            # ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            # ".reg .b64 smem_desc_b;\n\t"
            f".reg .b64 smem_desc_b_<{num_k_tile}>;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            # f"mov.b32 idesc, {hex(idesc)};\n\t"
            # f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            f"mov.b32 tmem_acc, $2;\n\t"
            "mov.b32 smem_desc_b_lo_start, $0;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, {smem_var_name_prefix}_0;\n\t"
            f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
            f"mov.b64 {smem_var_name_prefix}_0, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b_0, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            + "".join(
                (
                    f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, {smem_var_name_prefix}_{k};\n\t"
                    f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
                    f"add.s32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 {smem_var_name_prefix}_{k}, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b_{k}, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                )
                for k in range(1, num_k_tile)
            )
            + "setp.ne.b32 p, $1, 0;\n\t"
            # f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::f16 [tmem_acc], {smem_var_name_prefix}_0, smem_desc_b, idesc, {pred_str};\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], {smem_var_name_prefix}_0, smem_desc_b_0, {idesc_var_name}, {pred_str};\n\t"
            + "".join(
                (
                    # f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, {smem_var_name_prefix}_{k};\n\t"
                    # f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
                    # f"add.s32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    # f"mov.b64 {smem_var_name_prefix}_{k}, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    # f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    # f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::f16 [tmem_acc], {smem_var_name_prefix}_{k}, smem_desc_b, idesc, 1;\n\t"
                    # f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::f16 [tmem_acc], {smem_var_name_prefix}_{k}, smem_desc_b, {idesc_var_name}, 1;\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} [tmem_acc], {smem_var_name_prefix}_{k}, smem_desc_b_{k}, {idesc_var_name}, 1;\n\t"
                )
                for k in range(1, num_k_tile)
            )
            + "}\n",
            "r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
