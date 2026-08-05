# SPDX-License-Identifier: BSD-3-Clause

"""Parameterized gfx950 MXFP8 scaled GEMM.

E4M3 operands with per-32-element E8M0 block scales, folded into the CDNA4
scaled 16x16x128 MFMA. A is [M, K] and B is [N, K], both row-major over K;
the output is [M, N] bf16/fp16.

The tiling mirrors the FP16/BF16 gfx950 GEMM in ``gemm_gfx950.py``:

  * LDS blocking      -- a workgroup stages [block_m, block_k] of A and
                         [block_n, block_k] of B into LDS so every wave in the
                         group reuses them.
  * multi-stage       -- ``stages`` LDS buffers are filled by global->LDS DMA;
    pipelining           the loop waits on a counted ``vmcnt`` so the DMAs for
                         later stages stay in flight across the MFMAs.
  * register blocking -- each wave keeps mma_m_repeat x mma_n_repeat f32[4]
                         accumulators and reuses one LDS read across the repeat
                         loops.

Together these lift arithmetic intensity from the previous baseline's fixed
8 flops/byte (one wave, one 16x16 tile, operands straight from global) to
2*block_m*block_n/(block_m+block_n), which is what lets throughput scale with
problem size instead of saturating.

The MFMA operand plumbing follows FlyDSL's own tiled MX kernel
(``kernels/gemm/mxfp4_preshuffle.py``) rather than the CuTe-style
``make_fragment_A``/``make_tiled_mma`` path used by the FP16 kernel: the scaled
MFMA takes two extra i32 scale operands, which that path does not carry, so A/B
fragments are built by hand as rank-1 i32[8] registers.
"""

import functools
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec


MXFP8_SCALE_BLOCK_K = 32
MXFP8_MFMA_M = 16
MXFP8_MFMA_N = 16
MXFP8_MFMA_K = 128
GFX950_WAVE_SIZE = 64
GFX950_DMA_BYTES = 16
GFX950_LDS_CAPACITY = 163840
GFX950_MAX_BLOCK_THREADS = 1024
# Per-wave register blocking depth. Each k step issues Rm + Rn LDS reads and
# Rm * Rn MFMAs, so the cap directly sets the MFMA-per-LDS-read ratio that this
# kernel is limited by -- 4x4 gives 2, 8x8 gives 4. It is not an LDS-bandwidth
# limit (removing the XOR swizzle, i.e. going from 2-way to 16-way bank
# conflicts, costs only 1.5%) nor an occupancy limit (20 waves/CU at 2x2
# repeats is slower than 8 waves/CU at 4x4); the MFMA unit idles waiting on
# ds_read issue slots and latency, which deeper blocking amortizes.
#
# Measured on gfx950 with bf16 output: 8x8 uses 224 of 512 VGPRs with no
# scratch and is 1.17-1.29x faster than 4x4 for M >= 2048, while 16x8 spills
# and collapses to roughly a fifth of the throughput. The FP16 template caps at
# 4 because its 2-byte elements make each fragment twice as wide; e4m3 halves
# that, so the cutoff has to be re-derived rather than inherited.
MXFP8_MAX_MMA_REPEAT = 8
# 16-byte granules spanned by one 128-K MFMA step.
MXFP8_GRANULES_PER_MFMA_K = MXFP8_MFMA_K // GFX950_DMA_BYTES
# The f8f6f4 ABI splits a lane's 32 operand bytes into two 16-byte halves that
# sit 64 bytes apart inside the 128-K block.
MXFP8_HI_GRANULE_OFFSET = 4


@dataclass(frozen=True)
class MXFP8GemmParams:
    """Compile-time identity of one specialized MXFP8 kernel."""

    m: int
    n: int
    k: int
    out_dtype: str
    block_m: int = 128
    block_n: int = 128
    block_k: int = 256
    stages: int = 2
    m_waves: int = 2
    n_waves: int = 2
    group_m: int = 0

    def __cache_signature__(self):
        return (
            "mxfp8_gfx950_v2",
            self.m,
            self.n,
            self.k,
            self.out_dtype,
            self.block_m,
            self.block_n,
            self.block_k,
            self.stages,
            self.m_waves,
            self.n_waves,
            self.group_m,
        )


@dataclass(frozen=True)
class MXFP8GemmDerived:
    """Quantities derived from a tile config, shared by the kernel and the
    heuristics filter so the two can never disagree."""

    block_threads: int
    wave_tile_m: int
    wave_tile_n: int
    mma_m_repeat: int
    mma_n_repeat: int
    k_halves: int
    scale_cols: int
    granules_per_row: int
    ldg_a_iters: int
    ldg_b_iters: int
    ldg_wait_count: int
    a_stage_bytes: int
    b_stage_bytes: int
    smem_bytes: int


def mxfp8_gemm_derived(
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int = 0,
) -> MXFP8GemmDerived:
    """Validate a tile config and return its derived quantities.

    Raises ValueError for any config the kernel cannot express. Mirrors
    make_gemm_gfx950_param in the FP16 kernel.
    """
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise ValueError("block_m, block_n, and block_k must be positive")
    if stages < 2:
        raise ValueError("stages must be at least 2 for the staged LDS pipeline")
    if m_waves <= 0 or n_waves <= 0:
        raise ValueError("m_waves and n_waves must be positive")
    if group_m < 0:
        raise ValueError("group_m must be non-negative")
    if block_k % MXFP8_MFMA_K != 0:
        raise ValueError(
            f"block_k must be a multiple of the MFMA K depth: block_k={block_k}"
        )

    block_threads = m_waves * n_waves * GFX950_WAVE_SIZE
    if block_threads > GFX950_MAX_BLOCK_THREADS:
        raise ValueError(f"block exceeds {GFX950_MAX_BLOCK_THREADS} threads")

    wave_tile_m, rem_m = divmod(block_m, m_waves)
    wave_tile_n, rem_n = divmod(block_n, n_waves)
    if rem_m or rem_n:
        raise ValueError("block_m/block_n must be divisible by m_waves/n_waves")

    mma_m_repeat, rem_m = divmod(wave_tile_m, MXFP8_MFMA_M)
    mma_n_repeat, rem_n = divmod(wave_tile_n, MXFP8_MFMA_N)
    if rem_m or rem_n or mma_m_repeat == 0 or mma_n_repeat == 0:
        raise ValueError(
            "each wave tile must be a positive multiple of the 16x16 MFMA tile"
        )
    if mma_m_repeat > MXFP8_MAX_MMA_REPEAT or mma_n_repeat > MXFP8_MAX_MMA_REPEAT:
        raise ValueError(
            "accumulator repeats exceed the register budget: "
            f"mma_m_repeat={mma_m_repeat}, mma_n_repeat={mma_n_repeat}"
        )

    granules_per_row = block_k // GFX950_DMA_BYTES
    # The LDS layout XORs the granule index with the row, so the granule count
    # has to be a power of two or the swizzle would leave the row.
    if granules_per_row & (granules_per_row - 1):
        raise ValueError(
            f"block_k / {GFX950_DMA_BYTES} must be a power of two for the XOR "
            f"swizzle: block_k={block_k}"
        )

    dma_bytes_per_pass = block_threads * GFX950_DMA_BYTES
    if (block_m * block_k) % dma_bytes_per_pass != 0:
        raise ValueError(
            "A tile load schedule must exactly cover the LDS tile: "
            f"block_m={block_m}, block_k={block_k}, block_threads={block_threads}"
        )
    if (block_n * block_k) % dma_bytes_per_pass != 0:
        raise ValueError(
            "B tile load schedule must exactly cover the LDS tile: "
            f"block_n={block_n}, block_k={block_k}, block_threads={block_threads}"
        )
    ldg_a_iters = (block_m * block_k) // dma_bytes_per_pass
    ldg_b_iters = (block_n * block_k) // dma_bytes_per_pass
    ldg_wait_count = ldg_a_iters + ldg_b_iters

    a_stage_bytes = block_m * block_k
    b_stage_bytes = block_n * block_k
    smem_bytes = stages * (a_stage_bytes + b_stage_bytes)
    if smem_bytes > GFX950_LDS_CAPACITY:
        raise ValueError(
            "staged LDS buffers exceed the device shared-memory capacity: "
            f"stages={stages}, block_m={block_m}, block_n={block_n}, "
            f"block_k={block_k}, smem_bytes={smem_bytes}, "
            f"capacity={GFX950_LDS_CAPACITY}"
        )
    if (stages - 2) * ldg_wait_count >= 63:
        raise ValueError("staged pipeline wait count exceeds supported range")

    return MXFP8GemmDerived(
        block_threads=block_threads,
        wave_tile_m=wave_tile_m,
        wave_tile_n=wave_tile_n,
        mma_m_repeat=mma_m_repeat,
        mma_n_repeat=mma_n_repeat,
        k_halves=block_k // MXFP8_MFMA_K,
        scale_cols=block_k // MXFP8_SCALE_BLOCK_K,
        granules_per_row=granules_per_row,
        ldg_a_iters=ldg_a_iters,
        ldg_b_iters=ldg_b_iters,
        ldg_wait_count=ldg_wait_count,
        a_stage_bytes=a_stage_bytes,
        b_stage_bytes=b_stage_bytes,
        smem_bytes=smem_bytes,
    )


def make_mxfp8_param_and_validate(m, n, k, out_dtype, gemm_config):
    """Return MXFP8GemmParams for a concrete shape, or None if unsupported.

    Mirrors make_gemm_param_and_validate in the FP16 kernel: the autotuning
    filter uses None to mean "drop this choice".
    """
    if out_dtype not in ("bfloat16", "float16"):
        return None
    block_m = int(gemm_config["TILE_M"])
    block_n = int(gemm_config["TILE_N"])
    block_k = int(gemm_config["TILE_K"])
    stages = int(gemm_config["STAGES"])
    m_waves = int(gemm_config["BLOCK_M_WARPS"])
    n_waves = int(gemm_config["BLOCK_N_WARPS"])
    group_m = int(gemm_config["GROUP_M"])
    try:
        derived = mxfp8_gemm_derived(
            block_m, block_n, block_k, stages, m_waves, n_waves, group_m
        )
    except Exception:
        return None
    # No boundary predication: the tile must divide the problem exactly.
    if m % block_m or n % block_n or k % block_k:
        return None
    # The prologue fills stages-1 buffers before the steady-state loop runs.
    if (k // block_k) < stages:
        return None
    del derived
    return MXFP8GemmParams(
        m=m,
        n=n,
        k=k,
        out_dtype=out_dtype,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
    )


def make_mxfp8_gemm_kernel_name(param: MXFP8GemmParams) -> str:
    return (
        "mxfp8_scaled_mm_gfx950"
        f"_{param.out_dtype}"
        f"_bm{param.block_m}_bn{param.block_n}_bk{param.block_k}"
        f"_s{param.stages}_mw{param.m_waves}_nw{param.n_waves}"
        f"_g{param.group_m}"
    )


# TODO: Move common ROCm synchronization and buffer-load helpers to FlyDSL.
def __barrier(vmcnt=0):
    llvm.InlineAsmOp(
        None,
        [],
        f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier",
        "",
        has_side_effects=True,
    )


def buffer_load_lds_inline(rsrc, lds_ptr, global_offset, dma_bytes):
    buffer_load_asm_dict = {
        16: "buffer_load_dwordx4",
        8: "buffer_load_dwordx2",
        4: "buffer_load_dword",
    }
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
        f"s_mov_b32 m0, $0\n\t{buffer_load_asm_dict[dma_bytes]} $1, $2, 0 offen sc0 lds",
        "s,v,s",
        has_side_effects=True,
    )


@functools.lru_cache(maxsize=256)
def make_mxfp8_scaled_mm_gfx950(
    *,
    m: int,
    n: int,
    k: int,
    out_dtype: str,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 256,
    stages: int = 2,
    m_waves: int = 2,
    n_waves: int = 2,
    group_m: int = 0,
):
    """Build a tiled gfx950 MXFP8 scaled GEMM launcher for one tile config."""
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")
    d = mxfp8_gemm_derived(
        block_m, block_n, block_k, stages, m_waves, n_waves, group_m
    )
    if m % block_m or n % block_n or k % block_k:
        raise ValueError(
            f"shape must be divisible by the tile: {m}x{n}x{k} vs "
            f"{block_m}x{block_n}x{block_k}"
        )
    k_tiles = k // block_k
    if k_tiles < stages:
        raise ValueError("K must supply at least `stages` tiles for the pipeline")

    if out_dtype == "bfloat16":
        out_elem = fx.BFloat16
    elif out_dtype == "float16":
        out_elem = fx.Float16
    else:
        raise ValueError(f"unsupported MXFP8 output dtype: {out_dtype}")

    block_threads = d.block_threads
    granules_per_row = d.granules_per_row
    granule_mask = granules_per_row - 1
    scale_k = k // MXFP8_SCALE_BLOCK_K
    tiles_m = m // block_m
    tiles_n = n // block_n
    grid_size = tiles_m * tiles_n
    main_loop_end = k_tiles - (stages - 1)
    # Group consecutive workgroups into GROUP_M rows of the N sweep so their B
    # tiles stay hot in L2. Only exact groupings are used, which keeps the
    # index math free of a runtime min.
    use_group_m = group_m > 0 and tiles_m % group_m == 0 and tiles_m > group_m

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def kernel(
        a: fx.Tensor,
        b_nk: fx.Tensor,
        scale_a_u8: fx.Tensor,
        scale_b_u8: fx.Tensor,
        out: fx.Tensor,
    ):
        sa_u8 = fx.recast_iter(fx.Uint8, fx.get_iter(scale_a_u8))
        sb_u8 = fx.recast_iter(fx.Uint8, fx.get_iter(scale_b_u8))
        out_ptr = fx.recast_iter(out_elem, fx.get_iter(out))

        a_rsrc = fx.rocdl.get_buffer_rsrc(
            fx.get_iter(fx.rocdl.make_buffer_tensor(a, max_size=True))
        )
        b_rsrc = fx.rocdl.get_buffer_rsrc(
            fx.get_iter(fx.rocdl.make_buffer_tensor(b_nk, max_size=True))
        )

        tid = fx.Int32(fx.thread_idx.x)
        wave = rocdl.readfirstlane(T.i32, tid // fx.Int32(GFX950_WAVE_SIZE))
        lane = tid % fx.Int32(GFX950_WAVE_SIZE)
        lane_16 = lane % fx.Int32(MXFP8_MFMA_N)
        k_group = lane // fx.Int32(MXFP8_MFMA_N)
        wave_m = wave // fx.Int32(n_waves)
        wave_n = wave % fx.Int32(n_waves)

        pid = fx.Int32(fx.block_idx.x)
        if const_expr(use_group_m):
            group_tiles = group_m * tiles_n
            group_id = pid // fx.Int32(group_tiles)
            within = pid % fx.Int32(group_tiles)
            bid_m = group_id * fx.Int32(group_m) + within % fx.Int32(group_m)
            bid_n = within // fx.Int32(group_m)
        else:
            bid_m = pid // fx.Int32(tiles_n)
            bid_n = pid % fx.Int32(tiles_n)
        m_base = bid_m * fx.Int32(block_m)
        n_base = bid_n * fx.Int32(block_n)

        @fx.struct
        class SharedAB:
            a: fx.Array[fx.Int8, stages * d.a_stage_bytes, 16]
            b: fx.Array[fx.Int8, stages * d.b_stage_bytes, 16]

        lds = fx.SharedAllocator().allocate(SharedAB).peek()
        sA_i8 = fx.recast_iter(fx.Int8, lds.a.ptr)
        sB_i8 = fx.recast_iter(fx.Int8, lds.b.ptr)
        sA_i32 = fx.recast_iter(fx.Int32, lds.a.ptr)
        sB_i32 = fx.recast_iter(fx.Int32, lds.b.ptr)
        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

        mma_atom = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                MXFP8_MFMA_M,
                MXFP8_MFMA_N,
                MXFP8_MFMA_K,
                fx.Float8E4M3FN,
                fx.Float8E4M3FN,
                fx.Float32,
                opsel_a=0,
                opsel_b=0,
            )
        )

        def async_load_tile(rsrc, lds_i8, stage_bytes, ldg_iters, rows_base, k_tile, stage):
            """Stage one [rows, block_k] byte tile from global into LDS.

            The DMA writes lanes contiguously from a wave-uniform LDS base, so
            the destination is fixed and the *source* column is permuted to
            realize the XOR swizzle (same trick as the FP16 kernel).
            """
            stage_off = stage * fx.Int32(stage_bytes)
            for i in range_constexpr(ldg_iters):
                lin = (fx.Int32(i * block_threads) + tid) * fx.Int32(GFX950_DMA_BYTES)
                row = lin // fx.Int32(block_k)
                granule = (lin % fx.Int32(block_k)) // fx.Int32(GFX950_DMA_BYTES)
                src_granule = granule ^ (row & fx.Int32(granule_mask))
                gmem = (rows_base + row) * fx.Int32(k) + fx.Int32(
                    k_tile
                ) * fx.Int32(block_k) + src_granule * fx.Int32(GFX950_DMA_BYTES)
                wave_base = stage_off + fx.Int32(
                    (i * block_threads) * GFX950_DMA_BYTES
                ) + wave * fx.Int32(GFX950_WAVE_SIZE * GFX950_DMA_BYTES)
                lds_ptr = fx.add_offset(
                    lds_i8, rocdl.readfirstlane(T.i32, wave_base)
                )
                buffer_load_lds_inline(rsrc, lds_ptr, gmem, GFX950_DMA_BYTES)

        def async_load_a(k_tile, stage):
            async_load_tile(
                a_rsrc, sA_i8, d.a_stage_bytes, d.ldg_a_iters, m_base, k_tile, stage
            )

        def async_load_b(k_tile, stage):
            async_load_tile(
                b_rsrc, sB_i8, d.b_stage_bytes, d.ldg_b_iters, n_base, k_tile, stage
            )

        def read_frag(lds_i32, stage_bytes, stage, row_in_block, kh):
            """ds_read_b128 x2 -> one i32[8] MFMA operand fragment."""
            row_swz = row_in_block & fx.Int32(granule_mask)
            lo_logical = fx.Int32(kh * MXFP8_GRANULES_PER_MFMA_K) + k_group
            hi_logical = lo_logical + fx.Int32(MXFP8_HI_GRANULE_OFFSET)
            base_i32 = (
                stage * fx.Int32(stage_bytes // 4)
                + row_in_block * fx.Int32(block_k // 4)
            )

            def read16(logical):
                phys = logical ^ row_swz
                off = base_i32 + phys * fx.Int32(GFX950_DMA_BYTES // 4)
                frag = fx.make_rmem_tensor(4, fx.Int32)
                fx.copy(
                    lds_copy,
                    fx.make_view(fx.add_offset(lds_i32, off), fx.make_layout(4, 1)),
                    frag,
                )
                return frag

            lo = Vec(fx.memref_load_vec(read16(lo_logical)))
            hi = Vec(fx.memref_load_vec(read16(hi_logical)))
            frag = fx.make_rmem_tensor(8, fx.Int32)
            frag.store(lo.shuffle(hi, list(range(8))))
            return frag

        def load_scale_word(scale_u8, row_global, k_tile, kh):
            off = (
                fx.Int64(row_global) * fx.Int64(scale_k)
                + fx.Int64(k_tile) * fx.Int64(d.scale_cols)
                + fx.Int64(kh * (MXFP8_MFMA_K // MXFP8_SCALE_BLOCK_K))
                + fx.Int64(k_group)
            )
            scale_byte = fx.ptr_load(scale_u8 + off)
            # The atom reads one e8m0 byte per 32-K block; broadcasting keeps
            # it independent of the opsel byte lane.
            return scale_byte.to(fx.Int32) * fx.Int32(0x01010101)

        # Row/col of this wave's first 16x16 tile inside the block tile.
        wave_row0 = wave_m * fx.Int32(d.wave_tile_m)
        wave_col0 = wave_n * fx.Int32(d.wave_tile_n)

        accs = [
            fx.make_rmem_tensor(4, fx.Float32)
            for _ in range_constexpr(d.mma_m_repeat * d.mma_n_repeat)
        ]
        for idx in range_constexpr(d.mma_m_repeat * d.mma_n_repeat):
            accs[idx].store(Vec.filled(4, 0.0, fx.Float32))

        def compute_stage(stage, k_tile):
            for kh in range_constexpr(d.k_halves):
                a_frags = []
                sa_words = []
                for mi in range_constexpr(d.mma_m_repeat):
                    row_in_block = wave_row0 + fx.Int32(mi * MXFP8_MFMA_M) + lane_16
                    a_frags.append(
                        read_frag(sA_i32, d.a_stage_bytes, stage, row_in_block, kh)
                    )
                    sa_words.append(
                        load_scale_word(sa_u8, m_base + row_in_block, k_tile, kh)
                    )
                b_frags = []
                sb_words = []
                for ni in range_constexpr(d.mma_n_repeat):
                    col_in_block = wave_col0 + fx.Int32(ni * MXFP8_MFMA_N) + lane_16
                    b_frags.append(
                        read_frag(sB_i32, d.b_stage_bytes, stage, col_in_block, kh)
                    )
                    sb_words.append(
                        load_scale_word(sb_u8, n_base + col_in_block, k_tile, kh)
                    )
                # Register blocking: each LDS read feeds every repeat on the
                # other axis.
                for ni in range_constexpr(d.mma_n_repeat):
                    for mi in range_constexpr(d.mma_m_repeat):
                        acc = accs[mi * d.mma_n_repeat + ni]
                        fx.gemm(
                            mma_atom,
                            acc,
                            a_frags[mi],
                            b_frags[ni],
                            acc,
                            scale_a=sa_words[mi],
                            scale_b=sb_words[ni],
                        )

        # Prologue: fill stages-1 buffers so the steady-state loop always has a
        # landed tile to consume.
        for stage in range_constexpr(stages - 1):
            async_load_b(stage, fx.Int32(stage))
            async_load_a(stage, fx.Int32(stage))
        rocdl.sched_barrier(0)

        steady_wait = (stages - 2) * d.ldg_wait_count
        for kt in range(0, main_loop_end, 1):
            k_tile = fx.Int32(kt)
            cur = k_tile % fx.Int32(stages)
            write = (cur + fx.Int32(stages - 1)) % fx.Int32(stages)
            __barrier(steady_wait)
            async_load_b(k_tile + fx.Int32(stages - 1), write)
            async_load_a(k_tile + fx.Int32(stages - 1), write)
            compute_stage(cur, k_tile)

        # Drain: consume the buffers still in flight, walking vmcnt down to 0.
        for s in range_constexpr(stages - 1):
            k_tile = fx.Int32(main_loop_end + s)
            cur = k_tile % fx.Int32(stages)
            __barrier((stages - 2 - s) * d.ldg_wait_count)
            compute_stage(cur, k_tile)

        # Epilogue: each lane owns 4 rows of every 16x16 accumulator.
        for mi in range_constexpr(d.mma_m_repeat):
            row0 = (
                m_base
                + wave_row0
                + fx.Int32(mi * MXFP8_MFMA_M)
                + k_group * fx.Int32(4)
            )
            for ni in range_constexpr(d.mma_n_repeat):
                col = n_base + wave_col0 + fx.Int32(ni * MXFP8_MFMA_N) + lane_16
                out4 = Vec(accs[mi * d.mma_n_repeat + ni].load().ir_value()).to(
                    out_elem
                )
                for i in range_constexpr(4):
                    off = fx.Int64(row0 + fx.Int32(i)) * fx.Int64(n) + fx.Int64(col)
                    fx.ptr_store(out4[i], out_ptr + off)

    kernel._func.__name__ = make_mxfp8_gemm_kernel_name(
        MXFP8GemmParams(
            m=m,
            n=n,
            k=k,
            out_dtype=out_dtype,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            stages=stages,
            m_waves=m_waves,
            n_waves=n_waves,
            group_m=group_m,
        )
    )

    @flyc.jit
    def launch(
        a: fx.Tensor,
        b_nk: fx.Tensor,
        scale_a_u8: fx.Tensor,
        scale_b_u8: fx.Tensor,
        out: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(a, b_nk, scale_a_u8, scale_b_u8, out).launch(
            grid=(grid_size, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch
