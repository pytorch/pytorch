# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch


GFX950_DMA_BYTES = 16
GFX950_SCALE_DMA_BYTES = 4
GFX950_WAVE_SIZE = 64
_LDS_BANK_PERIOD_LOG2 = 6
_LDS_READ_B128_BASE = 3
_LDS_READ_TR16_BASE = 4

GEMM_DTYPE_FP32 = 1
GEMM_DTYPE_BF16 = 2
GEMM_DTYPE_FP16 = 3
GEMM_DTYPE_MXFP8 = 4
GEMM_DTYPE_MXFP4 = 5


GEMM_DTYPE_MMA = {
    GEMM_DTYPE_BF16: (16, 16, 32),
    GEMM_DTYPE_FP16: (16, 16, 32),
    GEMM_DTYPE_MXFP8: (16, 16, 128),
    GEMM_DTYPE_MXFP4: (16, 16, 128),
}
GEMM_DTYPE_BITS = {
    GEMM_DTYPE_FP32: 32,
    GEMM_DTYPE_BF16: 16,
    GEMM_DTYPE_FP16: 16,
    GEMM_DTYPE_MXFP8: 8,
    GEMM_DTYPE_MXFP4: 4,
}
MXFP_SCALE_BLOCK_K = 32
MXFP_MAX_MMA_REPEAT = 8


@fx.struct
class GemmGfx950Param:
    in_dtype_id: fx.Constexpr[int]
    out_dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    stages: fx.Constexpr[int]
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    group_m: fx.Constexpr[int]
    use_half_tile_interleaved: fx.Constexpr[bool]
    a_is_transposed: fx.Constexpr[bool]
    b_is_transposed: fx.Constexpr[bool]
    has_bias: fx.Constexpr[bool]
    has_k_tail: fx.Constexpr[bool]
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    cshuffle_dtype_id: fx.Constexpr[int]
    async_load_bytes: fx.Constexpr[int]
    in_data_bits: fx.Constexpr[int]
    out_data_bits: fx.Constexpr[int]
    cshuffle_r2g_vec_size: fx.Constexpr[int]
    ldg_x_threads: fx.Constexpr[int]
    block_threads: fx.Constexpr[int]
    ldg_a_iters: fx.Constexpr[int]
    ldg_b_iters: fx.Constexpr[int]
    # MXFP E8M0 scales staged through LDS.
    sa_stage_bytes: fx.Constexpr[int]
    sb_stage_bytes: fx.Constexpr[int]
    ldg_sa_iters: fx.Constexpr[int]
    ldg_sb_iters: fx.Constexpr[int]
    scale_row_bytes: fx.Constexpr[int]
    scale_chunk_tiles: fx.Constexpr[int]


@dataclass(slots=True, kw_only=True, eq=False)
class GemmABLoadContext:
    wave_offset: Any
    tid: Any
    k: Any
    param: GemmGfx950Param
    uni_copy_atom: Any
    buffer_copy_atom: Any
    a_s2r_copy_atom: Any
    b_s2r_copy_atom: Any
    thr_copy_a: Any
    thr_copy_b: Any


@dataclass(slots=True, kw_only=True, eq=False)
class AsyncLoadOperand:
    context: GemmABLoadContext
    src_base: Any
    lds_layout: Any
    outer_tile_size: Any
    outer_bound: Any
    leading_stride: Any
    load_iters: Any
    is_k_major: Any


def mxfp_scale_padded_bytes(rows, block_k, block_threads):
    workgroup_bytes = block_threads * GFX950_SCALE_DMA_BYTES
    n_bytes = rows * (block_k // MXFP_SCALE_BLOCK_K)
    return (n_bytes + workgroup_bytes - 1) // workgroup_bytes * workgroup_bytes


def make_gemm_gfx950_param(
    dtype_id: int | None = None,
    tile_m: int | None = None,
    tile_n: int | None = None,
    tile_k: int | None = None,
    stages: int = 2,
    m_waves: int = 2,
    n_waves: int = 4,
    group_m: int = 0,
    use_half_tile_interleaved: bool = False,
    a_is_transposed: bool = False,
    b_is_transposed: bool = True,
    has_bias: bool = False,
    has_k_tail: bool = False,
    *,
    out_dtype_id: int | None = None,
) -> GemmGfx950Param:
    # Keep the kernel implementation's internal block terminology unchanged.
    block_m, block_n, block_k = tile_m, tile_n, tile_k
    in_dtype_id = dtype_id
    if in_dtype_id not in (
        GEMM_DTYPE_BF16,
        GEMM_DTYPE_FP16,
        GEMM_DTYPE_MXFP8,
        GEMM_DTYPE_MXFP4,
    ):
        raise ValueError(f"unsupported in_dtype_id={in_dtype_id}")
    is_mxfp = in_dtype_id in (GEMM_DTYPE_MXFP8, GEMM_DTYPE_MXFP4)
    if block_m <= 0 or block_n <= 0 or block_k <= 0 or stages <= 0:
        raise ValueError("block_m, block_n, block_k, and stages must be positive")
    mma_m, mma_n, mma_k = GEMM_DTYPE_MMA[in_dtype_id]
    if stages < 2:
        raise ValueError("stages must be at least 2 for the staged LDS pipeline")
    if m_waves <= 0 or n_waves <= 0:
        raise ValueError("m_waves, and n_waves must be positive")
    if m_waves * n_waves > 16:
        raise ValueError("the workgroup cannot contain more than 16 waves")
    if group_m < 0:
        raise ValueError("group_m must be non-negative")
    if out_dtype_id is None:
        out_dtype_id = GEMM_DTYPE_BF16 if is_mxfp else in_dtype_id
    if out_dtype_id not in (
        (GEMM_DTYPE_BF16, GEMM_DTYPE_FP16) if is_mxfp else (in_dtype_id,)
    ):
        raise ValueError(f"unsupported out_dtype_id={out_dtype_id}")
    in_data_bits = GEMM_DTYPE_BITS[in_dtype_id]
    out_data_bits = GEMM_DTYPE_BITS[out_dtype_id]
    cshuffle_dtype_id = out_dtype_id
    block_k_bytes = block_k * in_data_bits // 8
    block_threads = m_waves * n_waves * GFX950_WAVE_SIZE
    max_cshuffle_r2g_vec_size = GFX950_DMA_BYTES * 8 // out_data_bits
    if use_half_tile_interleaved:
        half_block_m = block_m // 2
        half_block_n = block_n // 2
        if stages != 2:
            raise ValueError("half-tile interleaved kernel requires stages=2")
        if m_waves != 2 or n_waves < 2:
            raise ValueError(
                "half-tile interleaved kernel requires m_waves=2, n_waves>=2"
            )
        if half_block_m * 2 != block_m or half_block_n * 2 != block_n:
            raise ValueError(
                "half-tile interleaved kernel requires even block_m and block_n"
            )
        mma_m_half_repeat = half_block_m // m_waves // mma_m
        mma_n_half_repeat = half_block_n // n_waves // mma_n
        if (
            mma_m_half_repeat * m_waves * mma_m != half_block_m
            or mma_n_half_repeat * n_waves * mma_n != half_block_n
        ):
            raise ValueError("half tiles must be divisible by waves * MMA dimensions")
        if mma_n_half_repeat != 2:
            raise ValueError("HTI requires half_block_n / n_waves / mma_n == 2")
        stg_size_per_m_step = m_waves * mma_m * half_block_n
        if stg_size_per_m_step % block_threads != 0:
            raise ValueError("C-shuffle M step must be divisible by block_threads")
        stg_work_size_per_m_step = stg_size_per_m_step // block_threads
        cshuffle_r2g_vec_size = min(max_cshuffle_r2g_vec_size, stg_work_size_per_m_step)
        if (
            cshuffle_r2g_vec_size not in (4, 8)
            or stg_work_size_per_m_step % cshuffle_r2g_vec_size != 0
            or half_block_n % cshuffle_r2g_vec_size != 0
        ):
            raise ValueError(
                "half-tile C-shuffle must be covered by 4- or 8-element vectors"
            )
    else:
        cshuffle_r2g_vec_size = max_cshuffle_r2g_vec_size
        if block_n % cshuffle_r2g_vec_size != 0:
            raise ValueError("block_n must be divisible by the C-shuffle vector size")

    if is_mxfp:
        if block_k % MXFP_SCALE_BLOCK_K != 0:
            raise ValueError("MXFP block_k must be divisible by the scale block size")
        scale_rows_a = block_m // 2 if use_half_tile_interleaved else block_m
        scale_rows_b = block_n // 2 if use_half_tile_interleaved else block_n
        workgroup_bytes = block_threads * GFX950_SCALE_DMA_BYTES
        if use_half_tile_interleaved:
            # Fill at least one DMA pass for both halves, sharing an even K-tile count.
            # The smaller half determines the chunk; the larger may need more passes.
            double_tile_scale_bytes = (
                2 * min(scale_rows_a, scale_rows_b) * block_k // MXFP_SCALE_BLOCK_K
            )
            scale_chunk_tiles = 2 * (
                (workgroup_bytes + double_tile_scale_bytes - 1)
                // double_tile_scale_bytes
            )
        else:
            scale_chunk_tiles = 1
        scale_block_k = block_k * scale_chunk_tiles
        # Each allocation is one half in one scale slot; HTI has two chunk slots.
        sa_stage_bytes = mxfp_scale_padded_bytes(
            scale_rows_a, scale_block_k, block_threads
        )
        sb_stage_bytes = mxfp_scale_padded_bytes(
            scale_rows_b, scale_block_k, block_threads
        )
        # HTI chunk DMA is fenced separately, so it is not part of per-tile AB waits.
        ldg_sa_iters = (
            0 if use_half_tile_interleaved else sa_stage_bytes // workgroup_bytes
        )
        ldg_sb_iters = (
            0 if use_half_tile_interleaved else sb_stage_bytes // workgroup_bytes
        )
        scale_row_bytes = scale_block_k // MXFP_SCALE_BLOCK_K
    else:
        sa_stage_bytes = sb_stage_bytes = scale_chunk_tiles = 0
        ldg_sa_iters = ldg_sb_iters = scale_row_bytes = 0

    scale_halves = 2 if use_half_tile_interleaved else 1
    smem_bytes = stages * (
        block_m + block_n
    ) * block_k_bytes + stages * scale_halves * (sa_stage_bytes + sb_stage_bytes)
    smem_bytes = max(
        smem_bytes, block_m * block_n * GEMM_DTYPE_BITS[cshuffle_dtype_id] // 8
    )
    arch = get_rocm_arch()
    SMEM_CAPACITY_MAP = {
        "gfx942": 65536,
        "gfx950": 163840,
    }
    if arch not in SMEM_CAPACITY_MAP:
        raise ValueError(f"unsupported ROCm architecture: {arch}")
    smem_capacity = SMEM_CAPACITY_MAP[arch]
    if smem_bytes > smem_capacity:
        raise ValueError(
            "staged LDS buffers exceed the device shared-memory capacity: "
            f"stages={stages}, block_m={block_m}, block_n={block_n}, "
            f"block_k={block_k}, smem_bytes={smem_bytes}, "
            f"capacity={smem_capacity} for arch={arch}"
        )

    # async load check
    async_load_vec_size = GFX950_DMA_BYTES * 8 // in_data_bits
    ldg_x_threads = block_k // async_load_vec_size
    if ldg_x_threads * async_load_vec_size != block_k:
        raise ValueError(
            "block_k must be divisible by the async load vector size: "
            f"block_k={block_k}, async_load_vec_size={async_load_vec_size}, "
            f"covered_k={ldg_x_threads * async_load_vec_size}"
        )
    ldg_y_threads = block_threads // ldg_x_threads
    if ldg_y_threads * ldg_x_threads != block_threads:
        raise ValueError(
            "ldg thread layout must exactly cover the workgroup: "
            f"ldg_y_threads={ldg_y_threads}, ldg_x_threads={ldg_x_threads}, "
            f"block_threads={block_threads}"
        )
    ldg_a_iters = (block_m * block_k) // (block_threads * async_load_vec_size)
    ldg_b_iters = (block_n * block_k) // (block_threads * async_load_vec_size)
    if use_half_tile_interleaved:
        for operand, rows in (("A", block_m // 2), ("B", block_n // 2)):
            half_load_iters = rows * block_k // (block_threads * async_load_vec_size)
            if half_load_iters * block_threads * async_load_vec_size != rows * block_k:
                raise ValueError(
                    f"Half-tile {operand} must be covered by whole-thread vector loads: "
                    f"rows={rows}, block_k={block_k}, block_threads={block_threads}, "
                    f"async_load_vec_size={async_load_vec_size}"
                )
    if ldg_a_iters * block_threads * async_load_vec_size != block_m * block_k:
        raise ValueError(
            "A async load tile must be exactly covered by whole-thread vector loads: "
            f"block_m={block_m}, block_k={block_k}, block_threads={block_threads}, "
            f"async_load_vec_size={async_load_vec_size}"
        )
    if ldg_b_iters * block_threads * async_load_vec_size != block_n * block_k:
        raise ValueError(
            "B async load tile must be exactly covered by whole-thread vector loads: "
            f"block_n={block_n}, block_k={block_k}, block_threads={block_threads}, "
            f"async_load_vec_size={async_load_vec_size}"
        )
    if (stages - 2) * (ldg_a_iters + ldg_b_iters + ldg_sa_iters + ldg_sb_iters) >= 63:
        raise ValueError("staged pipeline vmcnt budget must be less than 63")
    # HTI keeps four half-tile accumulators live: budget the full output tile.
    mma_m_repeat = block_m // m_waves // mma_m
    mma_n_repeat = block_n // n_waves // mma_n
    mma_k_repeat = block_k // mma_k
    if mma_m_repeat * m_waves * mma_m != block_m:
        raise ValueError(
            "block_m must be divisible by m_waves * mma_m: "
            f"block_m={block_m}, m_waves={m_waves}, mma_m={mma_m}"
        )
    if mma_n_repeat * n_waves * mma_n != block_n:
        raise ValueError(
            "block_n must be divisible by n_waves * mma_n: "
            f"block_n={block_n}, n_waves={n_waves}, mma_n={mma_n}"
        )
    if is_mxfp and max(mma_m_repeat, mma_n_repeat) > MXFP_MAX_MMA_REPEAT:
        raise ValueError("accumulator repeats exceed the register budget")
    if mma_k_repeat * mma_k != block_k:
        raise ValueError(
            "block_k must be divisible by mma_k: "
            f"block_k={block_k}, mma_k={mma_k}, "
            f"mma_k_repeat={mma_k_repeat}, "
            f"covered_k={mma_k_repeat * mma_k}"
        )

    return GemmGfx950Param(
        in_dtype_id=in_dtype_id,
        out_dtype_id=out_dtype_id,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
        use_half_tile_interleaved=use_half_tile_interleaved,
        a_is_transposed=a_is_transposed,
        b_is_transposed=b_is_transposed,
        has_bias=has_bias,
        has_k_tail=has_k_tail,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
        cshuffle_dtype_id=cshuffle_dtype_id,
        async_load_bytes=GFX950_DMA_BYTES,
        in_data_bits=in_data_bits,
        out_data_bits=out_data_bits,
        cshuffle_r2g_vec_size=cshuffle_r2g_vec_size,
        ldg_x_threads=ldg_x_threads,
        block_threads=block_threads,
        ldg_a_iters=ldg_a_iters,
        ldg_b_iters=ldg_b_iters,
        sa_stage_bytes=sa_stage_bytes,
        sb_stage_bytes=sb_stage_bytes,
        ldg_sa_iters=ldg_sa_iters,
        ldg_sb_iters=ldg_sb_iters,
        scale_row_bytes=scale_row_bytes,
        scale_chunk_tiles=scale_chunk_tiles,
    )


def make_gemm_gfx950_kernel_name(param: GemmGfx950Param) -> str:
    dtype_names = {
        GEMM_DTYPE_FP32: "fp32",
        GEMM_DTYPE_BF16: "bf16",
        GEMM_DTYPE_FP16: "fp16",
        GEMM_DTYPE_MXFP8: "mxfp8",
        GEMM_DTYPE_MXFP4: "mxfp4",
    }
    dtype_str = dtype_names[param.in_dtype_id]
    out_suffix = (
        f"_{dtype_names[param.out_dtype_id]}"
        if param.out_dtype_id != param.in_dtype_id
        else ""
    )
    name = (
        f"gemm_{dtype_str}{out_suffix}_t{param.block_m}x{param.block_n}"
        f"x{param.block_k}x{param.stages}_ks1"
    )
    name += f"_w{param.m_waves}x{param.n_waves}x1"
    # Bias dtype is not part of the name; callers must keep it equal to out_dtype.
    name += f"_gm{param.group_m}_bias{int(param.has_bias)}"
    name += f"_ktail{int(param.has_k_tail)}"
    a_layout = "t" if param.a_is_transposed else "n"
    b_layout = "t" if param.b_is_transposed else "n"
    name += f"_l{a_layout}{b_layout}"
    name += "_phti" if param.use_half_tile_interleaved else "_pft"
    return name


class BlockSwizzle:
    def __init__(self, NUM_XCDS, NUM_PIDS_THRESHOLD, GROUP_M, N_MAJOR_FALLBACK=False):
        self.NUM_XCDS = NUM_XCDS
        self.NUM_PIDS_THRESHOLD = NUM_PIDS_THRESHOLD
        self.GROUP_M = GROUP_M
        self.N_MAJOR_FALLBACK = N_MAJOR_FALLBACK

    @flyc.jit
    def swizzle(self, num_pid_m, num_pid_n, pid):
        if const_expr(self.N_MAJOR_FALLBACK):
            simple_m = pid % num_pid_m
            simple_n = pid // num_pid_m
        else:
            simple_m = pid // num_pid_n
            simple_n = pid % num_pid_n
        if const_expr(self.GROUP_M <= 0):
            return simple_m, simple_n
        num_xcds = self.NUM_XCDS
        swizzle_threshold = self.NUM_PIDS_THRESHOLD
        num_wg = num_pid_m * num_pid_n
        linear_id = pid
        intra_xcd = linear_id // num_xcds
        xcd = linear_id % num_xcds
        wgid = xcd * (num_wg // num_xcds) + intra_xcd
        group_m = self.GROUP_M
        wgid_per_group = group_m * num_pid_n
        group_id = wgid // wgid_per_group
        intra_group = wgid % wgid_per_group
        first_pid_m = group_id * group_m
        remaining_m = num_pid_m - first_pid_m
        group_size_m = (remaining_m < group_m).select(remaining_m, group_m)
        swizzled_n = intra_group // group_size_m
        swizzled_m = first_pid_m + (intra_group % group_size_m)
        use_simple = (num_wg < swizzle_threshold) | ((num_wg % num_xcds) != 0)
        if const_expr(isinstance(use_simple, bool)):
            if const_expr(use_simple):
                return simple_m, simple_n
            return swizzled_m, swizzled_n
        return (
            use_simple.select(simple_m, swizzled_m),
            use_simple.select(simple_n, swizzled_n),
        )


def _make_xor_swizzle(mask, base, shift):
    return fx.static(fx.SwizzleType.get(mask, base, shift))


def make_lds_layout(rows, block_k, is_transposed):
    if const_expr(is_transposed):
        contiguous_extent = rows
        base = _LDS_READ_TR16_BASE
        order = (0, 1)
    else:
        contiguous_extent = block_k
        base = _LDS_READ_B128_BASE
        order = (1, 0)

    base_layout = fx.make_ordered_layout((rows, block_k), order)
    extent_log2 = contiguous_extent.bit_length() - 1
    mask = _LDS_BANK_PERIOD_LOG2 - base
    shift = extent_log2 - base
    is_power_of_two = contiguous_extent == 1 << extent_log2
    if const_expr(not is_power_of_two or shift < mask):
        return base_layout
    return fx.make_composed_layout(
        _make_xor_swizzle(mask, base, shift),
        base_layout,
    )


def make_gemm_ab_lds_layouts(rows_a, rows_b, block_k, a_is_transposed, b_is_transposed):
    return (
        make_lds_layout(rows_a, block_k, a_is_transposed),
        make_lds_layout(rows_b, block_k, not b_is_transposed),
    )


def get_wave_lds_offset(tid, async_load_bytes):
    return rocdl.readfirstlane(
        fx.Int64.ir_type,
        fx.Int64(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * async_load_bytes),
    )


def make_wave_lds_ptr(ptr, wave_offset):
    return fx.recast_iter(fx.Int8, ptr) + fx.Int32(wave_offset)


def swizzled_contiguous_idx(idx0, idx1, layout, extent):
    # The XOR swizzle is self-inverse. Map each physical contiguous position
    # written by direct-to-LDS DMA back to its logical global vector.
    elem_offset = fx.get_scalar(fx.crd2idx((idx0, idx1), layout))
    return elem_offset % extent


# TODO: Move common ROCm synchronization and buffer-load helpers to FlyDSL.
def __barrier(vmcnt=0):
    llvm.InlineAsmOp(
        None,
        [],
        f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier",
        "",
        has_side_effects=True,
    )


def __waitcnt(vmcnt=0):
    llvm.InlineAsmOp(None, [], f"s_waitcnt vmcnt({vmcnt})", "", has_side_effects=True)


def buffer_load_lds_inline(rsrc, lds_ptr, global_offset, dma_bytes):
    buffer_load_asm_dict = {
        16: "buffer_load_dwordx4",
        8: "buffer_load_dwordx2",
        4: "buffer_load_dword",
    }
    # Match LLVM's gfx950 buffer_load_lds lowering: VMEM needs one wait state
    # after the SALU write to M0 (llvm-project#116681).
    llvm.InlineAsmOp(
        None,
        [
            llvm.IntToPtrOp(
                ir.Type.parse("!llvm.ptr<3>"),
                fx.as_ir_value(fx.ptrtoint(lds_ptr)),
            ).result,
            fx.as_ir_value(global_offset),
            fx.as_ir_value(rsrc),
        ],
        f"s_mov_b32 m0, $0\n\ts_nop 0\n\t{buffer_load_asm_dict[dma_bytes]} $1, $2, 0 offen sc0 lds",
        "s,v,s",
        has_side_effects=True,
    )


def _operand_fragment_dtype(param: GemmGfx950Param):
    # MXFP4 remains packed: each 8-bit carrier fragment holds two FP4 values.
    # The actual scaled MMA atom still uses Float4E2M1FN.
    if const_expr(param.in_dtype_id in (GEMM_DTYPE_MXFP4, GEMM_DTYPE_MXFP8)):
        return fx.Float8E4M3FN
    return (
        fx.Float16 if const_expr(param.in_dtype_id == GEMM_DTYPE_FP16) else fx.BFloat16
    )


def make_gemm_ab_load_context(
    operand_fragment_dtype,
    tiled_mma,
    tid,
    k,
    param: GemmGfx950Param,
):
    uni_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), operand_fragment_dtype)
    buffer_copy_atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy128b(), operand_fragment_dtype
    )

    lds_read = (
        fx.rocdl.cdna4.LDSReadTrans8_64b
        if param.in_dtype_id in (GEMM_DTYPE_MXFP4, GEMM_DTYPE_MXFP8)
        else fx.rocdl.cdna4.LDSReadTrans16_64b
    )
    if const_expr(param.a_is_transposed):
        a_s2r_copy_atom = fx.make_copy_atom(lds_read(), operand_fragment_dtype)
        a_tiled_copy_atom = a_s2r_copy_atom
    else:
        a_s2r_copy_atom = uni_copy_atom
        a_tiled_copy_atom = buffer_copy_atom
    if const_expr(not param.b_is_transposed):
        b_s2r_copy_atom = fx.make_copy_atom(lds_read(), operand_fragment_dtype)
        b_tiled_copy_atom = b_s2r_copy_atom
    else:
        b_s2r_copy_atom = uni_copy_atom
        b_tiled_copy_atom = buffer_copy_atom

    return GemmABLoadContext(
        wave_offset=get_wave_lds_offset(tid, param.async_load_bytes),
        tid=tid,
        k=k,
        param=param,
        uni_copy_atom=uni_copy_atom,
        buffer_copy_atom=buffer_copy_atom,
        a_s2r_copy_atom=a_s2r_copy_atom,
        b_s2r_copy_atom=b_s2r_copy_atom,
        thr_copy_a=fx.make_tiled_copy_A(a_tiled_copy_atom, tiled_mma).get_slice(tid),
        thr_copy_b=fx.make_tiled_copy_B(b_tiled_copy_atom, tiled_mma).get_slice(tid),
    )


def async_load_operand(
    operand: AsyncLoadOperand,
    lds_base,
    global_outer_offset,
    k_tile,
):
    context = operand.context
    param = context.param
    # Pin the measured scaled-MMA load schedule across inline asm; this is
    # a compiler scheduling constraint, not a memory fence or workgroup barrier.
    pin_load_schedule = param.in_dtype_id in (GEMM_DTYPE_MXFP4, GEMM_DTYPE_MXFP8)
    tid = context.tid
    block_threads = param.block_threads
    in_data_bits = param.in_data_bits
    async_load_vec_size = param.async_load_bytes * 8 // in_data_bits
    src_bits = operand.src_base.dtype.width
    ldg_x_threads = param.ldg_x_threads
    block_k = param.block_k
    # This PyTorch kernel has neither split-K nor slice-K, so each workgroup
    # starts at logical K=0 and only the tile index contributes to the address.
    k_begin = k_tile * (block_k * in_data_bits // src_bits)
    k_bound = context.k * in_data_bits // src_bits
    lds_ptr = fx.recast_iter(fx.Uint8, lds_base) + fx.Int32(context.wave_offset)
    rsrc = fx.rocdl.get_buffer_rsrc(operand.src_base)
    for i in range_constexpr(operand.load_iters):
        global_tid = block_threads * i + tid
        if const_expr(operand.is_k_major):
            outer_vec_size = param.async_load_bytes * 8 // src_bits
            outer_x_threads = operand.outer_tile_size // outer_vec_size
            outer_lds_idx = global_tid % outer_x_threads * outer_vec_size
            k_local_idx = global_tid // outer_x_threads
            outer_local_idx = swizzled_contiguous_idx(
                outer_lds_idx,
                k_local_idx,
                operand.lds_layout,
                operand.outer_tile_size,
            )
            global_k_idx = k_begin + k_local_idx
        else:
            outer_local_idx = global_tid // ldg_x_threads
            k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
            global_k_idx = k_begin + swizzled_contiguous_idx(
                outer_local_idx,
                k_local_idx * in_data_bits // src_bits,
                operand.lds_layout,
                block_k * in_data_bits // src_bits,
            )
        if const_expr(param.has_k_tail):
            safe_global_k_idx = (global_k_idx < k_bound).select(global_k_idx, 0)
        else:
            safe_global_k_idx = global_k_idx
        global_outer_idx = global_outer_offset + outer_local_idx
        safe_global_outer_idx = (global_outer_idx < operand.outer_bound).select(
            global_outer_idx, 0
        )
        if const_expr(operand.is_k_major):
            global_offset = (
                safe_global_k_idx * operand.leading_stride + safe_global_outer_idx
            )
        else:
            global_offset = (
                safe_global_outer_idx * operand.leading_stride + safe_global_k_idx
            )
        if const_expr(pin_load_schedule):
            rocdl.sched_barrier(0)
        buffer_load_lds_inline(
            rsrc,
            lds_ptr,
            global_offset * (src_bits // 8),
            param.async_load_bytes,
        )
        if const_expr(pin_load_schedule):
            rocdl.sched_barrier(0)
        if i < operand.load_iters - 1:
            lds_ptr = lds_ptr + block_threads * param.async_load_bytes


def make_mxfp_lds_layout(rows, storage_k, is_transposed):
    if const_expr(is_transposed):
        return make_lds_layout(rows, storage_k, True)
    layout = fx.make_ordered_layout((rows, storage_k), (1, 0))
    # Preserve each packed K row and the low four bits of a 16-byte DMA.
    # S<3,4,4> needs 128-byte row alignment; e.g. FP4 BK384 only has 64.
    if const_expr(storage_k % 128 == 0):
        return fx.make_composed_layout(fx.static(fx.SwizzleType.get(3, 4, 4)), layout)
    if const_expr(storage_k % 64 == 0):
        return fx.make_composed_layout(fx.static(fx.SwizzleType.get(2, 4, 3)), layout)
    return layout


def make_gemm_tiled_mma(param: GemmGfx950Param):
    if const_expr(param.in_dtype_id == GEMM_DTYPE_MXFP8):
        op = rocdl.cdna4.MFMA_Scale(
            param.mma_m, param.mma_n, param.mma_k, fx.Float8E4M3FN
        )
    elif const_expr(param.in_dtype_id == GEMM_DTYPE_MXFP4):
        # TiledMma only supplies layouts here.  A byte fragment carries two
        # packed FP4 values; compute itself uses the scaled FP4 atom below.
        op = rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k // 2, fx.Float8E4M3FN)
    else:
        dtype = (
            fx.Float16
            if const_expr(param.in_dtype_id == GEMM_DTYPE_FP16)
            else fx.BFloat16
        )
        op = rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, dtype)

    permutation = None
    if const_expr(param.in_dtype_id == GEMM_DTYPE_MXFP8):
        permutation = fx.make_tile(None, None, fx.make_layout((16, 2, 4), (1, 64, 16)))
    return fx.make_tiled_mma(
        fx.make_mma_atom(op),
        fx.make_layout(
            (param.m_waves, param.n_waves, 1),
            (param.n_waves, 1, 0),
        ),
        permutation,
    )


def make_mxfp_mma_atom(param: GemmGfx950Param):
    dtype = (
        fx.Float4E2M1FN
        if const_expr(param.in_dtype_id == GEMM_DTYPE_MXFP4)
        else fx.Float8E4M3FN
    )
    return fx.make_mma_atom(
        rocdl.cdna4.MFMA_Scale(param.mma_m, param.mma_n, param.mma_k, dtype)
    )


def async_load_mxfp_scales(
    scale,
    lds_base,
    tid,
    outer_offset,
    outer_bound,
    k_begin,
    rows,
    block_k,
    block_threads,
    has_k_tail,
    k_bound,
):
    """Stage four E8M0 bytes/lane with the same waitcnt protocol as A/B."""
    scale_k = block_k // MXFP_SCALE_BLOCK_K
    words_per_row = scale_k // 4
    stage_bytes = mxfp_scale_padded_bytes(rows, block_k, block_threads)
    load_iters = stage_bytes // (block_threads * GFX950_SCALE_DMA_BYTES)
    rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(scale))
    lds_ptr = make_wave_lds_ptr(
        lds_base, get_wave_lds_offset(tid, GFX950_SCALE_DMA_BYTES)
    )
    stride = fx.Int32(fx.get_scalar(scale.stride[0]))
    for i in range_constexpr(load_iters):
        word = i * block_threads + tid
        row = outer_offset + (word // words_per_row) % rows
        safe_row = (row < outer_bound).select(row, 0)
        sk = k_begin // MXFP_SCALE_BLOCK_K + word % words_per_row * 4
        if const_expr(has_k_tail):
            sk = (sk < k_bound // MXFP_SCALE_BLOCK_K).select(sk, 0)
        buffer_load_lds_inline(
            rsrc, lds_ptr, safe_row * stride + sk, GFX950_SCALE_DMA_BYTES
        )
        if i < load_iters - 1:
            lds_ptr = lds_ptr + block_threads * GFX950_SCALE_DMA_BYTES


def mxfp_gemm(
    frag_C,
    frag_A,
    frag_B,
    scale_a,
    scale_b,
    tid,
    m_offset,
    n_offset,
    k_offset,
    m,
    n,
    param: GemmGfx950Param,
    scales_are_fragments=False,
):
    """Apply one MMA K step with unshuffled [outer, K // 32] E8M0 scales.

    lane % mma_m selects the row/column and lane // mma_m selects its
    32-element K scale group. opsel=0 consumes the low byte of each
    lane's i32 scale operand.
    """
    mma_atom = make_mxfp_mma_atom(param)
    if const_expr(not scales_are_fragments):
        lane = tid % GFX950_WAVE_SIZE
        wave = tid // GFX950_WAVE_SIZE
        wave_m = wave // param.n_waves
        wave_n = wave % param.n_waves
        scale_lane = lane % param.mma_m
        scale_group = lane // param.mma_m
        scale_k = k_offset // MXFP_SCALE_BLOCK_K + scale_group

    a_scales = []
    for mi in range_constexpr(fx.size(frag_A.shape[1]).unpack()):
        if const_expr(scales_are_fragments):
            a_scales.append(scale_a[mi])
        else:
            row = m_offset + (mi * param.m_waves + wave_m) * param.mma_m + scale_lane
            safe_row = (row < m).select(row, 0)
            a_scales.append(scale_a[safe_row, scale_k].to(fx.Int32))
    b_scales = []
    for ni in range_constexpr(fx.size(frag_B.shape[1]).unpack()):
        if const_expr(scales_are_fragments):
            b_scales.append(scale_b[ni])
        else:
            col = n_offset + (ni * param.n_waves + wave_n) * param.mma_n + scale_lane
            safe_col = (col < n).select(col, 0)
            b_scales.append(scale_b[safe_col, scale_k].to(fx.Int32))

    for ni in range_constexpr(fx.size(frag_B.shape[1]).unpack()):
        for mi in range_constexpr(fx.size(frag_A.shape[1]).unpack()):
            a = fx.coalesce(frag_A[None, mi])
            b = fx.coalesce(frag_B[None, ni])
            sa, sb = a_scales[mi], b_scales[ni]
            fx.gemm(
                mma_atom,
                fx.coalesce(frag_C[None, mi, ni]),
                a,
                b,
                fx.coalesce(frag_C[None, mi, ni]),
                scale_a=sa,
                scale_b=sb,
            )


def make_ab_lds_layouts(rows_a, rows_b, storage_k, param):
    if const_expr(param.in_dtype_id in (GEMM_DTYPE_MXFP4, GEMM_DTYPE_MXFP8)):
        return (
            make_mxfp_lds_layout(rows_a, storage_k, param.a_is_transposed),
            make_mxfp_lds_layout(rows_b, storage_k, not param.b_is_transposed),
        )
    return make_gemm_ab_lds_layouts(
        rows_a, rows_b, storage_k, param.a_is_transposed, param.b_is_transposed
    )


@flyc.kernel
def gemm_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmGfx950Param,
    # Empty tuples have no device ABI operands for the unscaled dtypes.
    scale_a=(),
    scale_b=(),
):
    is_mxfp = param.in_dtype_id in (GEMM_DTYPE_MXFP4, GEMM_DTYPE_MXFP8)
    block_k_storage = param.block_k // (
        2 if param.in_dtype_id == GEMM_DTYPE_MXFP4 else 1
    )
    cshuffle_dtype = (
        fx.BFloat16 if param.out_dtype_id == GEMM_DTYPE_BF16 else fx.Float16
    )
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    stages = param.stages
    has_k_tail = param.has_k_tail
    block_threads = param.block_threads
    ldg_a_iters = param.ldg_a_iters
    ldg_b_iters = param.ldg_b_iters
    ldg_wait_count = ldg_a_iters + ldg_b_iters + param.ldg_sa_iters + param.ldg_sb_iters
    operand_fragment_dtype = _operand_fragment_dtype(param)

    tid = fx.thread_idx.x
    num_pid_m = (m - 1) // block_m + 1
    num_pid_n = (n - 1) // block_n + 1
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    )
    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    k_tiles = (k - 1) // block_k + 1

    if const_expr(is_mxfp):

        @fx.struct
        class SharedABStorage:
            a: fx.Array[operand_fragment_dtype, stages * block_m * block_k_storage, 16]
            b: fx.Array[operand_fragment_dtype, stages * block_n * block_k_storage, 16]
            sa: fx.Array[fx.Uint8, stages * param.sa_stage_bytes, 16]
            sb: fx.Array[fx.Uint8, stages * param.sb_stage_bytes, 16]
    else:

        @fx.struct
        class SharedABStorage:
            a: fx.Array[operand_fragment_dtype, stages * block_m * block_k_storage, 16]
            b: fx.Array[operand_fragment_dtype, stages * block_n * block_k_storage, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[cshuffle_dtype, block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr
    if const_expr(is_mxfp):
        smem_sa = storage.ab.sa.peek().ptr
        smem_sb = storage.ab.sb.peek().ptr
        scale_a_buf = fx.rocdl.make_buffer_tensor(scale_a, max_size=True)
        scale_b_buf = fx.rocdl.make_buffer_tensor(scale_b, max_size=True)

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)
    if const_expr(param.has_bias):
        bias_buf = fx.rocdl.make_buffer_tensor(bias, max_size=True)
    else:
        bias_buf = None

    gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]
    thr_mma = tiled_mma.thr_slice(tid)
    ab_load_context = make_gemm_ab_load_context(
        operand_fragment_dtype, tiled_mma, tid, k, param
    )
    uni_copy_atom = (
        fx.make_copy_atom(fx.UniversalCopy128b(), cshuffle_dtype)
        if is_mxfp
        else ab_load_context.uni_copy_atom
    )
    buffer_copy_atom = (
        fx.make_copy_atom(fx.rocdl.BufferCopy128b(), cshuffle_dtype)
        if is_mxfp
        else ab_load_context.buffer_copy_atom
    )
    a_s2r_copy_atom = ab_load_context.a_s2r_copy_atom
    b_s2r_copy_atom = ab_load_context.b_s2r_copy_atom
    thr_copy_A = ab_load_context.thr_copy_a
    thr_copy_B = ab_load_context.thr_copy_b
    a_lds_layout, b_lds_layout = make_ab_lds_layouts(
        block_m, block_n, block_k_storage, param
    )
    a_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        src_base=fx.get_iter(a_buf),
        lds_layout=a_lds_layout,
        outer_tile_size=block_m,
        outer_bound=m,
        leading_stride=a_leading_stride,
        load_iters=ldg_a_iters,
        is_k_major=param.a_is_transposed,
    )
    b_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        src_base=fx.get_iter(b_buf),
        lds_layout=b_lds_layout,
        outer_tile_size=block_n,
        outer_bound=n,
        leading_stride=b_leading_stride,
        load_iters=ldg_b_iters,
        is_k_major=not param.b_is_transposed,
    )
    c_lds_layout = fx.make_layout((block_m, block_n), (block_n, 1))

    sA = fx.make_view(smem_a, a_lds_layout)
    sB = fx.make_view(smem_b, b_lds_layout)
    sC = fx.make_view(smem_c, c_lds_layout)

    frag_A = thr_mma.make_fragment_A(sA)
    frag_B = thr_mma.make_fragment_B(sB)
    frag_C = thr_mma.make_fragment_C(gC)
    frag_A_retile = thr_copy_A.retile(frag_A)
    frag_B_retile = thr_copy_B.retile(frag_B)

    row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
    thr_mma_cRow = thr_mma.partition_C(row_coords)
    thr_mma_cCol = thr_mma.partition_C(col_coords)

    cshuffle_vec_size = param.cshuffle_r2g_vec_size
    cshuffle_x_threads = block_n // cshuffle_vec_size
    cshuffle_thr_layout = fx.make_layout(
        (block_threads // cshuffle_x_threads, cshuffle_x_threads),
        (cshuffle_x_threads, 1),
    )
    cshuffle_val_layout = fx.make_layout((1, cshuffle_vec_size), (1, 1))
    cshuffle_tile, cshuffle_tv_layout = fx.make_layout_tv(
        cshuffle_thr_layout,
        cshuffle_val_layout,
    )
    tiled_copy_cshuffle = fx.make_tiled_copy(
        buffer_copy_atom,
        cshuffle_tv_layout,
        cshuffle_tile,
    )
    thr_copy_cshuffle = tiled_copy_cshuffle.get_slice(tid)
    thr_sC = thr_copy_cshuffle.partition_S(sC)
    thr_gC = thr_copy_cshuffle.partition_D(gC)
    thr_cRow = thr_copy_cshuffle.partition_S(row_coords)[(0, None), None, None]
    thr_cCol = thr_copy_cshuffle.partition_S(col_coords)[(0, None), None, None]
    frag_C_cshuffle = fx.make_fragment_like(thr_sC)
    pred_C = fx.make_fragment_like(thr_cRow, dtype=fx.Boolean)

    frag_C.fill(0.0)
    if const_expr(param.has_bias and not is_mxfp):
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            col_idx = fx.get_scalar(thr_mma_cCol[i])
            global_n_idx = bid_n * block_n + col_idx
            safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
            frag_C[i] = bias_buf[safe_global_n_idx].to(fx.Float32)

    for i in range_constexpr(fx.size(pred_C.shape).unpack()):
        local_row = fx.get_scalar(thr_cRow[i])
        local_col = fx.get_scalar(thr_cCol[i])
        row_idx = bid_m * block_m + local_row
        col_idx = bid_n * block_n + local_col
        pred_C[i] = (
            (local_row < block_m)
            & (local_col < block_n)
            & (row_idx < m)
            & (col_idx < n)
        )

    def async_load_a_to_lds(k_tile, stage):
        async_load_operand(
            a_load_operand,
            lds_base=smem_a + stage * block_m * block_k_storage,
            global_outer_offset=bid_m * block_m,
            k_tile=k_tile,
        )
        if const_expr(is_mxfp):
            async_load_mxfp_scales(
                scale_a_buf,
                smem_sa + stage * param.sa_stage_bytes,
                tid,
                bid_m * block_m,
                m,
                k_tile * block_k,
                block_m,
                block_k,
                block_threads,
                has_k_tail,
                k,
            )

    def async_load_b_to_lds(k_tile, stage):
        async_load_operand(
            b_load_operand,
            lds_base=smem_b + stage * block_n * block_k_storage,
            global_outer_offset=bid_n * block_n,
            k_tile=k_tile,
        )
        if const_expr(is_mxfp):
            async_load_mxfp_scales(
                scale_b_buf,
                smem_sb + stage * param.sb_stage_bytes,
                tid,
                bid_n * block_n,
                n,
                k_tile * block_k,
                block_n,
                block_k,
                block_threads,
                has_k_tail,
                k,
            )

    def compute_stage(read_stage, k_tile):
        sA_stage = fx.make_view(
            smem_a + read_stage * block_m * block_k_storage, a_lds_layout
        )
        sB_stage = fx.make_view(
            smem_b + read_stage * block_n * block_k_storage, b_lds_layout
        )
        thr_sA_s2r = thr_copy_A.partition_S(sA_stage)
        thr_sB_s2r = thr_copy_B.partition_S(sB_stage)
        if const_expr(is_mxfp):
            scale_a_view = fx.make_view(
                smem_sa + read_stage * param.sa_stage_bytes,
                fx.make_layout(
                    (block_m, param.scale_row_bytes), (param.scale_row_bytes, 1)
                ),
            )
            scale_b_view = fx.make_view(
                smem_sb + read_stage * param.sb_stage_bytes,
                fx.make_layout(
                    (block_n, param.scale_row_bytes), (param.scale_row_bytes, 1)
                ),
            )

        def compute_k_chunk(block_k_iter):
            fx.copy(
                b_s2r_copy_atom,
                thr_sB_s2r[None, None, block_k_iter],
                frag_B_retile[None, None, block_k_iter],
            )
            fx.copy(
                a_s2r_copy_atom,
                thr_sA_s2r[None, None, block_k_iter],
                frag_A_retile[None, None, block_k_iter],
            )
            if const_expr(is_mxfp):
                mxfp_gemm(
                    frag_C,
                    frag_A[None, None, block_k_iter],
                    frag_B[None, None, block_k_iter],
                    scale_a_view,
                    scale_b_view,
                    tid,
                    0,
                    0,
                    block_k_iter * param.mma_k,
                    block_m,
                    block_n,
                    param,
                )
            else:
                fx.gemm(
                    tiled_mma,
                    frag_C,
                    frag_A[None, None, block_k_iter],
                    frag_B[None, None, block_k_iter],
                    frag_C,
                    traversal_order=fx.GemmTraversalOrder.KNM,
                )

        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    compute_k_chunk(block_k_iter)
            else:
                compute_k_chunk(block_k_iter)

    for stage in range_constexpr(stages - 1):
        async_load_b_to_lds(stage, stage)
        async_load_a_to_lds(stage, stage)
    rocdl.sched_barrier(0)

    if const_expr(has_k_tail):
        main_loop_end = (k_tiles > stages - 1).select(k_tiles - (stages - 1), 0)
    else:
        main_loop_end = k_tiles - (stages - 1)
    for k_tile in range(0, main_loop_end, 1):
        current_stage = k_tile % stages
        write_stage = (current_stage + stages - 1) % stages
        __barrier((stages - 2) * ldg_wait_count)
        async_load_b_to_lds(k_tile + (stages - 1), write_stage)
        async_load_a_to_lds(k_tile + (stages - 1), write_stage)
        compute_stage(current_stage, k_tile)

    current_stage = main_loop_end % stages
    for s in range_constexpr(0, stages - 1):
        __barrier((stages - 2 - s) * ldg_wait_count)
        compute_stage(current_stage, main_loop_end + s)
        current_stage = (current_stage + 1) % stages

    frag_C_out = fx.make_fragment_like(frag_C, cshuffle_dtype)
    if const_expr(is_mxfp):
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            acc = frag_C[i]
            if const_expr(param.has_bias):
                col = bid_n * block_n + fx.get_scalar(thr_mma_cCol[i])
                acc = acc + bias_buf[(col < n).select(col, 0)].to(fx.Float32)
            frag_C_out[i] = acc.to(cshuffle_dtype)
    else:
        frag_C_out.store(frag_C.load().to(cshuffle_dtype))

    fx.gpu.barrier()
    for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
        row = fx.get_scalar(thr_mma_cRow[i])
        col = fx.get_scalar(thr_mma_cCol[i])
        sC[row, col] = frag_C_out[i]

    fx.gpu.barrier()
    fx.copy(uni_copy_atom, thr_sC, frag_C_cshuffle)
    fx.copy(buffer_copy_atom, frag_C_cshuffle, thr_gC, pred=pred_C)


@flyc.kernel
def gemm_hti_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmGfx950Param,
    # Empty tuples have no device ABI operands for the unscaled dtypes.
    scale_a=(),
    scale_b=(),
):
    is_mxfp = param.in_dtype_id in (GEMM_DTYPE_MXFP4, GEMM_DTYPE_MXFP8)
    operand_fragment_dtype = _operand_fragment_dtype(param)
    elements_per_byte = 2 if const_expr(param.in_dtype_id == GEMM_DTYPE_MXFP4) else 1
    cshuffle_dtype = (
        fx.BFloat16 if const_expr(param.out_dtype_id == GEMM_DTYPE_BF16) else fx.Float16
    )
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    block_k_storage = block_k // elements_per_byte
    half_block_m = block_m // 2
    half_block_n = block_n // 2
    stages = param.stages
    has_k_tail = param.has_k_tail
    block_threads = param.block_threads
    n_waves = param.n_waves
    scale_chunk_tiles = param.scale_chunk_tiles if is_mxfp else 1
    use_scale_chunk = scale_chunk_tiles > 1
    scale_stages = stages
    scale_k = param.scale_row_bytes
    scale_a_stage_bytes = param.sa_stage_bytes
    scale_b_stage_bytes = param.sb_stage_bytes
    scale_a_iters = param.ldg_sa_iters
    scale_b_iters = param.ldg_sb_iters
    a_load_iters = param.ldg_a_iters // 2
    b_load_iters = param.ldg_b_iters // 2
    half_ldg_a_iters = a_load_iters + scale_a_iters
    half_ldg_b_iters = b_load_iters + scale_b_iters

    tid = fx.thread_idx.x
    wid = tid // GFX950_WAVE_SIZE
    num_pid_m = (m - 1) // block_m + 1
    num_pid_n = (n - 1) // block_n + 1
    bid_m, bid_n = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    ).swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    k_tiles = (k - 1) // block_k + 1
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

    if const_expr(is_mxfp):

        @fx.struct
        class SharedABStorage:
            a: fx.Array[operand_fragment_dtype, stages * block_m * block_k_storage, 16]
            b: fx.Array[operand_fragment_dtype, stages * block_n * block_k_storage, 16]
            sa: fx.Array[fx.Uint8, scale_stages * 2 * scale_a_stage_bytes, 16]
            sb: fx.Array[fx.Uint8, scale_stages * 2 * scale_b_stage_bytes, 16]
    else:

        @fx.struct
        class SharedABStorage:
            a: fx.Array[operand_fragment_dtype, stages * block_m * block_k_storage, 16]
            b: fx.Array[operand_fragment_dtype, stages * block_n * block_k_storage, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[
            cshuffle_dtype,
            (half_block_m * half_block_n if is_mxfp else block_m * block_n),
            16,
        ]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    if const_expr(is_mxfp):
        smem_sa = storage.ab.sa.peek().ptr
        smem_sb = storage.ab.sb.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=is_mxfp)
    if const_expr(is_mxfp):
        scale_a_buf = fx.rocdl.make_buffer_tensor(scale_a, max_size=True)
        scale_b_buf = fx.rocdl.make_buffer_tensor(scale_b, max_size=True)
    if const_expr(param.has_bias):
        bias_buf = fx.rocdl.make_buffer_tensor(bias, max_size=True)
    else:
        bias_buf = None

    ab_load_context = make_gemm_ab_load_context(
        operand_fragment_dtype, tiled_mma, tid, k, param
    )
    a_s2r_copy_atom = ab_load_context.a_s2r_copy_atom
    b_s2r_copy_atom = ab_load_context.b_s2r_copy_atom
    thr_copy_A = ab_load_context.thr_copy_a
    thr_copy_B = ab_load_context.thr_copy_b
    thr_mma = tiled_mma.thr_slice(tid)
    a_lds_layout, b_lds_layout = make_ab_lds_layouts(
        half_block_m, half_block_n, block_k_storage, param
    )
    a_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        src_base=fx.get_iter(a_buf),
        lds_layout=a_lds_layout,
        outer_tile_size=half_block_m,
        outer_bound=m,
        leading_stride=a_leading_stride,
        load_iters=a_load_iters,
        is_k_major=param.a_is_transposed,
    )
    b_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        src_base=fx.get_iter(b_buf),
        lds_layout=b_lds_layout,
        outer_tile_size=half_block_n,
        outer_bound=n,
        leading_stride=b_leading_stride,
        load_iters=b_load_iters,
        is_k_major=not param.b_is_transposed,
    )

    def half_a_base(stage, m_part):
        return smem_a + (stage * block_m + m_part * half_block_m) * block_k_storage

    def half_b_base(stage, n_part):
        return smem_b + (stage * block_n + n_part * half_block_n) * block_k_storage

    def async_load_half(part, k_tile, stage, is_a):
        operand, lds_base, outer_offset, rows = (
            (a_load_operand, half_a_base(stage, part), block_m_offset, half_block_m)
            if is_a
            else (
                b_load_operand,
                half_b_base(stage, part),
                block_n_offset,
                half_block_n,
            )
        )
        async_load_operand(
            operand,
            lds_base=lds_base,
            global_outer_offset=outer_offset + part * rows,
            k_tile=k_tile,
        )
        if const_expr(is_mxfp and not use_scale_chunk):
            issue_scale_piece(k_tile, part, is_a, stage)

    def async_load_a_to_lds(part, k_tile, stage):
        async_load_half(part, k_tile, stage, True)

    def async_load_b_to_lds(part, k_tile, stage):
        async_load_half(part, k_tile, stage, False)

    def issue_scale_piece(k_tile, part, is_a, slot=0):
        if const_expr(use_scale_chunk):
            slot = k_tile // scale_chunk_tiles % stages
        base, stage_bytes, buf, offset, bound, rows = (
            (smem_sa, scale_a_stage_bytes, scale_a_buf, block_m_offset, m, half_block_m)
            if is_a
            else (
                smem_sb,
                scale_b_stage_bytes,
                scale_b_buf,
                block_n_offset,
                n,
                half_block_n,
            )
        )
        async_load_mxfp_scales(
            buf,
            base + (slot * 2 + part) * stage_bytes,
            tid,
            offset + part * rows,
            bound,
            k_tile * block_k,
            rows,
            block_k * scale_chunk_tiles,
            block_threads,
            has_k_tail or use_scale_chunk,
            k,
        )

    def issue_scale_chunk(k_tile):
        for part in range_constexpr(2):
            issue_scale_piece(k_tile, part, True)
            issue_scale_piece(k_tile, part, False)

    def prefetch_scale_chunk(k_tile):
        if const_expr(use_scale_chunk):
            if k_tile % scale_chunk_tiles == 0:
                # Align staggered M-wave groups before recycling a chunk slot.
                rocdl.sched_barrier(0)
                if wid // n_waves == 0:
                    rocdl.s_barrier()
                __barrier(0)
                issue_scale_chunk(k_tile + scale_chunk_tiles)
                if wid // n_waves == 1:
                    rocdl.s_barrier()
                rocdl.sched_barrier(0)

    def make_gC(m_part, n_part):
        return fx.flat_divide(out_buf, (half_block_m, half_block_n))[
            None, None, bid_m * 2 + m_part, bid_n * 2 + n_part
        ]

    def make_c_fragment(m_part, n_part):
        frag_C = thr_mma.make_fragment_C(make_gC(m_part, n_part))
        frag_C.fill(0.0)
        return frag_C

    def load_scale_fragment(base, stage_bytes, part, stage, rows, waves, wave, k_tile):
        if const_expr(use_scale_chunk):
            stage = k_tile // scale_chunk_tiles % stages
            offset = k_tile % scale_chunk_tiles * (block_k // MXFP_SCALE_BLOCK_K)
        else:
            offset = 0
        scale_view = fx.make_view(
            base + (stage * 2 + part) * stage_bytes,
            fx.make_layout((rows, scale_k), (scale_k, 1)),
        )
        repeats = rows // waves // param.mma_m
        frag = fx.make_rmem_tensor((repeats, block_k // param.mma_k), fx.Int32)
        lane = tid % GFX950_WAVE_SIZE
        for ki in range_constexpr(block_k // param.mma_k):
            for ri in range_constexpr(repeats):
                row = (ri * waves + wave) * param.mma_m + lane % param.mma_m
                col = ki * (param.mma_k // MXFP_SCALE_BLOCK_K) + lane // param.mma_m
                frag[ri, ki] = scale_view[row, offset + col].to(fx.Int32)
        return frag

    def load_a_fragment(m_part, read_stage, k_tile):
        sA = fx.make_view(half_a_base(read_stage, m_part), a_lds_layout)
        frag_A = thr_mma.make_fragment_A(sA)
        frag_A_retile = thr_copy_A.retile(frag_A)
        thr_sA_s2r = thr_copy_A.partition_S(sA)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    fx.copy(
                        a_s2r_copy_atom,
                        thr_sA_s2r[None, None, block_k_iter],
                        frag_A_retile[None, None, block_k_iter],
                    )
            else:
                fx.copy(
                    a_s2r_copy_atom,
                    thr_sA_s2r[None, None, block_k_iter],
                    frag_A_retile[None, None, block_k_iter],
                )
        # Preserve scales with the data before prefetch reuses this LDS half.
        scales = None
        if const_expr(is_mxfp):
            scales = load_scale_fragment(
                smem_sa,
                scale_a_stage_bytes,
                m_part,
                read_stage,
                half_block_m,
                param.m_waves,
                wid // n_waves,
                k_tile,
            )
        return frag_A, scales

    def load_b_fragment(n_part, read_stage, k_tile):
        sB = fx.make_view(half_b_base(read_stage, n_part), b_lds_layout)
        frag_B = thr_mma.make_fragment_B(sB)
        frag_B_retile = thr_copy_B.retile(frag_B)
        thr_sB_s2r = thr_copy_B.partition_S(sB)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    fx.copy(
                        b_s2r_copy_atom,
                        thr_sB_s2r[None, None, block_k_iter],
                        frag_B_retile[None, None, block_k_iter],
                    )
            else:
                fx.copy(
                    b_s2r_copy_atom,
                    thr_sB_s2r[None, None, block_k_iter],
                    frag_B_retile[None, None, block_k_iter],
                )
        scales = None
        if const_expr(is_mxfp):
            scales = load_scale_fragment(
                smem_sb,
                scale_b_stage_bytes,
                n_part,
                read_stage,
                half_block_n,
                param.n_waves,
                wid % n_waves,
                k_tile,
            )
        return frag_B, scales

    def consume(k_tile, frag_C, frag_A, frag_B):
        frag_A, scales_a = frag_A
        frag_B, scales_b = frag_B
        rocdl.sched_barrier(0)

        def mma_k_chunk(block_k_iter):
            if const_expr(is_mxfp):
                mxfp_gemm(
                    frag_C,
                    frag_A[None, None, block_k_iter],
                    frag_B[None, None, block_k_iter],
                    scales_a[None, block_k_iter],
                    scales_b[None, block_k_iter],
                    tid,
                    0,
                    0,
                    block_k_iter * param.mma_k,
                    half_block_m,
                    half_block_n,
                    param,
                    scales_are_fragments=True,
                )
            else:
                fx.gemm(
                    tiled_mma,
                    frag_C,
                    frag_A[None, None, block_k_iter],
                    frag_B[None, None, block_k_iter],
                    frag_C,
                    traversal_order=fx.GemmTraversalOrder.KNM,
                )

        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    mma_k_chunk(block_k_iter)
            else:
                mma_k_chunk(block_k_iter)
        rocdl.sched_barrier(0)

    cshuffle_s2r_atom = fx.make_copy_atom(fx.UniversalCopy128b(), cshuffle_dtype)
    cshuffle_r2g_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), cshuffle_dtype)

    def cshuffle_views():
        sC = fx.make_view(
            smem_c, fx.make_layout((half_block_m, half_block_n), (half_block_n, 1))
        )

        row_coords = fx.make_view(
            0, fx.make_layout((half_block_m, half_block_n), (1, 0))
        )
        col_coords = fx.make_view(
            0, fx.make_layout((half_block_m, half_block_n), (0, 1))
        )
        thr_mma_cRow = thr_mma.partition_C(row_coords)
        thr_mma_cCol = thr_mma.partition_C(col_coords)

        cshuffle_vec_size = param.cshuffle_r2g_vec_size
        cshuffle_x_threads = half_block_n // cshuffle_vec_size
        cshuffle_thr_layout = fx.make_layout(
            (block_threads // cshuffle_x_threads, cshuffle_x_threads),
            (cshuffle_x_threads, 1),
        )
        cshuffle_val_layout = fx.make_layout((1, cshuffle_vec_size), (1, 1))
        cshuffle_tile, cshuffle_tv_layout = fx.make_layout_tv(
            cshuffle_thr_layout,
            cshuffle_val_layout,
        )
        tiled_copy_cshuffle = fx.make_tiled_copy(
            cshuffle_r2g_atom,
            cshuffle_tv_layout,
            cshuffle_tile,
        )
        thr_copy_cshuffle = tiled_copy_cshuffle.get_slice(tid)
        thr_sC = thr_copy_cshuffle.partition_S(sC)
        thr_cRow = thr_copy_cshuffle.partition_S(row_coords)[(0, None), None, None]
        thr_cCol = thr_copy_cshuffle.partition_S(col_coords)[(0, None), None, None]
        return (
            sC,
            thr_mma_cRow,
            thr_mma_cCol,
            thr_sC,
            thr_cRow,
            thr_cCol,
            thr_copy_cshuffle,
        )

    if const_expr(is_mxfp):
        mx_cshuffle_views = cshuffle_views()

    def store_half_tile(m_part, n_part, frag_C):
        if const_expr(is_mxfp):
            gC = make_gC(m_part, n_part)
        else:
            gC = fx.flat_divide(out_buf, (half_block_m, half_block_n))[
                None, None, bid_m * 2 + m_part, bid_n * 2 + n_part
            ]
        views = mx_cshuffle_views if is_mxfp else cshuffle_views()
        (
            sC,
            thr_mma_cRow,
            thr_mma_cCol,
            thr_sC,
            thr_cRow,
            thr_cCol,
            thr_copy_cshuffle,
        ) = views
        thr_gC = thr_copy_cshuffle.partition_D(gC)
        frag_C_cshuffle = fx.make_fragment_like(thr_sC)
        pred_C = fx.make_fragment_like(thr_cRow, dtype=fx.Boolean)

        s2r = cshuffle_s2r_atom
        r2g = cshuffle_r2g_atom
        for i in range_constexpr(fx.size(pred_C.shape).unpack()):
            local_row = fx.get_scalar(thr_cRow[i])
            local_col = fx.get_scalar(thr_cCol[i])
            row_idx = bid_m * block_m + m_part * half_block_m + local_row
            col_idx = bid_n * block_n + n_part * half_block_n + local_col
            pred_C[i] = (
                (local_row < half_block_m)
                & (local_col < half_block_n)
                & (row_idx < m)
                & (col_idx < n)
            )

        frag_C_out = fx.make_fragment_like(frag_C, cshuffle_dtype)
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            val = frag_C[i]
            if const_expr(param.has_bias):
                col = fx.get_scalar(thr_mma_cCol[i])
                global_n_idx = bid_n * block_n + n_part * half_block_n + col
                safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
                val = val + bias_buf[safe_global_n_idx].to(fx.Float32)
            frag_C_out[i] = val.to(cshuffle_dtype)

        fx.gpu.barrier()
        for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
            row = fx.get_scalar(thr_mma_cRow[i])
            col = fx.get_scalar(thr_mma_cCol[i])
            sC[row, col] = frag_C_out[i]

        fx.gpu.barrier()
        fx.copy(s2r, thr_sC, frag_C_cshuffle)
        fx.copy(
            r2g,
            frag_C_cshuffle,
            thr_gC,
            pred=pred_C,
        )
        fx.gpu.barrier()

    c00 = make_c_fragment(0, 0)
    c01 = make_c_fragment(0, 1)
    c10 = make_c_fragment(1, 0)
    c11 = make_c_fragment(1, 1)

    if const_expr(use_scale_chunk):
        issue_scale_chunk(0)
        __barrier(0)

    async_load_b_to_lds(0, 0, 0)
    async_load_a_to_lds(0, 0, 0)
    async_load_b_to_lds(1, 0, 0)
    async_load_a_to_lds(1, 0, 0)
    rocdl.sched_barrier(0)
    if wid // n_waves == 1:
        rocdl.s_barrier()
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    async_load_b_to_lds(0, 1, 1)
    async_load_a_to_lds(0, 1, 1)
    async_load_b_to_lds(1, 1, 1)
    __barrier(half_ldg_b_iters + half_ldg_a_iters)

    def compute_double_tile(k_tile, prefetch_next):
        prefetch_scale_chunk(k_tile)
        next_k_tile = k_tile + 2

        b0 = load_b_fragment(0, 0, k_tile)
        a0 = load_a_fragment(0, 0, k_tile)
        async_load_a_to_lds(1, k_tile + 1, 1)
        rocdl.s_barrier()
        consume(k_tile, c00, a0, b0)
        rocdl.s_barrier()

        b1 = load_b_fragment(1, 0, k_tile)
        if const_expr(prefetch_next):
            async_load_b_to_lds(0, next_k_tile, 0)
            rocdl.s_barrier()
        consume(k_tile, c01, a0, b1)
        rocdl.s_barrier()

        a1 = load_a_fragment(1, 0, k_tile)
        if const_expr(prefetch_next):
            async_load_a_to_lds(0, next_k_tile, 0)
            rocdl.s_barrier()
        consume(k_tile, c10, a1, b0)
        rocdl.s_barrier()

        b0 = load_b_fragment(0, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_b_to_lds(1, next_k_tile, 0)
            __barrier(2 * half_ldg_b_iters + half_ldg_a_iters)
        consume(k_tile, c11, a1, b1)
        if const_expr(not prefetch_next):
            __waitcnt(0)
        rocdl.s_barrier()

        a0 = load_a_fragment(0, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_a_to_lds(1, next_k_tile, 0)
            rocdl.s_barrier()
        consume(k_tile + 1, c00, a0, b0)
        rocdl.s_barrier()

        b1 = load_b_fragment(1, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_b_to_lds(0, next_k_tile + 1, 1)
            rocdl.s_barrier()
        consume(k_tile + 1, c01, a0, b1)
        rocdl.s_barrier()

        a1 = load_a_fragment(1, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_a_to_lds(0, next_k_tile + 1, 1)
            rocdl.s_barrier()
        consume(k_tile + 1, c10, a1, b0)
        rocdl.s_barrier()

        if const_expr(prefetch_next):
            async_load_b_to_lds(1, next_k_tile + 1, 1)
            __barrier(half_ldg_b_iters + half_ldg_a_iters)
        consume(k_tile + 1, c11, a1, b1)
        rocdl.s_barrier()

    final_double_tile = ((k_tiles % 2) == 0).select(k_tiles - 2, k_tiles - 1)
    main_loop_end = (k_tiles > 2).select(final_double_tile, 0)
    for k_tile in range(0, main_loop_end, 2):
        compute_double_tile(k_tile, True)

    compute_double_tile(main_loop_end, False)

    store_half_tile(0, 0, c00)
    store_half_tile(0, 1, c01)
    store_half_tile(1, 0, c10)
    store_half_tile(1, 1, c11)


def _launch_gemm(out, a, b, bias, param, stream, scale_a=(), scale_b=()):
    m = fx.Int32(fx.get_scalar(a.shape[0]))
    n = fx.Int32(fx.get_scalar(b.shape[1]))
    k = fx.Int32(fx.get_scalar(a.shape[1]))
    if const_expr(param.in_dtype_id == GEMM_DTYPE_MXFP4):
        k = k * fx.Int32(2)
    a_leading_stride = fx.Int32(
        fx.get_scalar(a.stride[1] if const_expr(param.a_is_transposed) else a.stride[0])
    )
    b_leading_stride = fx.Int32(
        fx.get_scalar(b.stride[1] if const_expr(param.b_is_transposed) else b.stride[0])
    )
    tiled_mma = make_gemm_tiled_mma(param)
    num_pid_m = (m - 1) // param.block_m + 1
    num_pid_n = (n - 1) // param.block_n + 1
    kernel_impl = (
        gemm_hti_gfx950_kernel
        if param.use_half_tile_interleaved
        else gemm_gfx950_kernel
    )
    kernel_impl._known_block_size = [param.block_threads, 1, 1]
    kernel_impl._func.__name__ = make_gemm_gfx950_kernel_name(param)
    kernel_impl(
        out,
        a,
        b,
        bias,
        m,
        n,
        k,
        a_leading_stride,
        b_leading_stride,
        tiled_mma,
        param,
        scale_a,
        scale_b,
    ).launch(
        grid=(num_pid_m * num_pid_n, 1, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


def infer_has_k_tail(
    k: int, tile_k: int, stages: int, use_half_tile_interleaved: bool = False
):
    # HTI consumes K tiles in pairs, so an odd count is also a tail.
    k_tiles = (k + tile_k - 1) // tile_k
    return (
        (k % tile_k != 0)
        or (k_tiles < stages - 1)
        or (use_half_tile_interleaved and k_tiles % 2 != 0)
    )


def make_gemm_param_and_validate(m, n, k, kwargs):
    try:
        result = make_gemm_gfx950_param(**kwargs)
    except ValueError:
        return None
    is_mxfp = result.in_dtype_id in (GEMM_DTYPE_MXFP4, GEMM_DTYPE_MXFP8)
    if is_mxfp and (
        min(m, n, k) <= 0
        or k > 2**31 - 1
        or (not result.b_is_transposed and n % GFX950_DMA_BYTES != 0)
    ):
        return None
    output_vec_size = GFX950_DMA_BYTES // (result.out_data_bits // 8)
    if n % output_vec_size != 0 or k % result.mma_k != 0:
        return None
    async_load_vec_size = GFX950_DMA_BYTES * 8 // max(8, result.in_data_bits)
    if result.a_is_transposed and m % async_load_vec_size != 0:
        return None
    if result.b_is_transposed and k % async_load_vec_size != 0:
        return None
    if result.use_half_tile_interleaved:
        k_tiles = (k + result.block_k - 1) // result.block_k
        if k_tiles < 2:
            return None
    return result


@flyc.jit
def gemm_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    param: GemmGfx950Param,
    stream: fx.Stream = fx.Stream(None),
):
    """Compute out[M, N] = a[M, K] @ b[K, N] with dynamic tensor layouts."""
    _launch_gemm(out, a, b, out, param, stream)


@flyc.jit
def gemm_mxfp_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    scale_a: fx.Tensor,
    scale_b: fx.Tensor,
    bias: fx.Tensor,
    param: GemmGfx950Param,
    stream: fx.Stream = fx.Stream(None),
):
    _launch_gemm(out, a, b, bias, param, stream, scale_a, scale_b)
