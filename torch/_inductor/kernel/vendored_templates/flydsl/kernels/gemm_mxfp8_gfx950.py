# SPDX-License-Identifier: BSD-3-Clause

"""Tile/Layout gfx950 MXFP8 scaled GEMM.

E4M3 operands use per-32-element E8M0 block scales in CDNA4 scaled
16x16x128 MFMA instructions. A is [M, K], B is [N, K], and both are row-major
over K. TiledMma and TiledCopy describe wave, fragment, and output ownership;
target-specific direct-to-LDS DMA and the staged waitcnt pipeline remain
explicit.
"""

import functools
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl


MXFP8_SCALE_BLOCK_K = 32
MXFP8_MFMA_M = 16
MXFP8_MFMA_N = 16
MXFP8_MFMA_K = 128
GFX950_WAVE_SIZE = 64
GFX950_DMA_BYTES = 16
GFX950_LDS_CAPACITY = 163840
GFX950_MAX_BLOCK_THREADS = 1024
# The pre-Layout v2 kernel benefited from 8x8 register blocking. Keep that
# search bound while v3 resource use is re-measured.
MXFP8_MAX_MMA_REPEAT = 8


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
    frag_first: int = 1

    def __cache_signature__(self):
        return (
            "mxfp8_gfx950_v3",
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
            self.frag_first,
        )


@dataclass(frozen=True)
class MXFP8GemmDerived:
    """Quantities derived from a tile config, shared by the kernel and the
    heuristics filter so the two can never disagree."""

    block_threads: int
    mma_m_repeat: int
    mma_n_repeat: int
    k_halves: int
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
        mma_m_repeat=mma_m_repeat,
        mma_n_repeat=mma_n_repeat,
        k_halves=block_k // MXFP8_MFMA_K,
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
    frag_first = int(gemm_config["FRAG_FIRST"])
    if frag_first not in (0, 1):
        return None
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
        frag_first=frag_first,
    )


def make_mxfp8_gemm_kernel_name(param: MXFP8GemmParams) -> str:
    return (
        "mxfp8_scaled_mm_gfx950"
        f"_{param.out_dtype}"
        f"_bm{param.block_m}_bn{param.block_n}_bk{param.block_k}"
        f"_s{param.stages}_mw{param.m_waves}_nw{param.n_waves}"
        f"_g{param.group_m}_f{param.frag_first}"
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
    frag_first: int = 1,
):
    """Build a tiled gfx950 MXFP8 scaled GEMM launcher for one tile config."""
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")
    d = mxfp8_gemm_derived(block_m, block_n, block_k, stages, m_waves, n_waves, group_m)
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
    swizzle_bits = granules_per_row.bit_length() - 1
    scale_k = k // MXFP8_SCALE_BLOCK_K
    tiles_m = m // block_m
    tiles_n = n // block_n
    grid_size = tiles_m * tiles_n
    main_loop_end = k_tiles - (stages - 1)
    # Group consecutive workgroups into GROUP_M rows of the N sweep so their B
    # tiles stay hot in L2. Only exact groupings are used, which keeps the
    # index math free of a runtime min.
    use_group_m = group_m > 0 and tiles_m % group_m == 0 and tiles_m > group_m
    # One dword feeds exactly four repeats through the 4x4 lane-group transpose,
    # so shallower register blocking keeps the per-byte scale loads.
    packed_scale = d.mma_m_repeat % 4 == 0 and d.mma_n_repeat % 4 == 0

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def kernel(
        a: fx.Tensor,
        b_nk: fx.Tensor,
        scale_a_u8: fx.Tensor,
        scale_b_u8: fx.Tensor,
        out: fx.Tensor,
    ):
        tid = fx.thread_idx.x

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
        class SharedStorage:
            a: fx.Array[fx.Float8E4M3FN, stages * block_m * block_k, 16]
            b: fx.Array[fx.Float8E4M3FN, stages * block_n * block_k, 16]

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        smem_a = storage.a.ptr
        smem_b = storage.b.ptr

        def make_flat_buffer(tensor, elems):
            flat = fx.Tensor(
                fx.make_view(fx.get_iter(tensor), fx.make_layout(elems, 1))
            )
            return fx.rocdl.make_buffer_tensor(flat, max_size=True)

        a_flat = fx.logical_divide(make_flat_buffer(a, m * k), fx.make_layout(1, 1))
        b_flat = fx.logical_divide(make_flat_buffer(b_nk, n * k), fx.make_layout(1, 1))
        sa_flat = fx.logical_divide(
            make_flat_buffer(scale_a_u8, m * scale_k), fx.make_layout(1, 1)
        )
        sb_flat = fx.logical_divide(
            make_flat_buffer(scale_b_u8, n * scale_k), fx.make_layout(1, 1)
        )
        out_view = fx.Tensor(
            fx.make_view(
                fx.get_iter(out),
                fx.make_layout((m, n), (n, 1)),
            )
        )
        out_buf = fx.rocdl.make_buffer_tensor(out_view, max_size=True)
        gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]

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
        mma_permutation = fx.make_tile(
            None,
            None,
            fx.make_layout(
                (GFX950_DMA_BYTES, 2, MXFP8_MFMA_K // (2 * GFX950_DMA_BYTES)),
                (1, MXFP8_MFMA_K // 2, GFX950_DMA_BYTES),
            ),
        )
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout(
                (m_waves, n_waves, 1),
                (n_waves, 1, 0),
            ),
            mma_permutation,
        )
        thr_mma = tiled_mma.thr_slice(tid)

        s2r_layout_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FN)
        s2r_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float8E4M3FN)
        thr_copy_A = fx.make_tiled_copy_A(s2r_layout_atom, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(s2r_layout_atom, tiled_mma).get_slice(tid)
        dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        scale_atom = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), fx.Uint8)

        swizzle = fx.static(fx.SwizzleType.get(swizzle_bits, 4, swizzle_bits))

        def make_lds_layout(rows):
            return fx.make_composed_layout(
                swizzle,
                fx.make_ordered_layout((rows, block_k), (1, 0)),
            )

        a_lds_layout = make_lds_layout(block_m)
        b_lds_layout = make_lds_layout(block_n)
        sA = fx.make_view(smem_a, a_lds_layout)
        sB = fx.make_view(smem_b, b_lds_layout)

        frag_A = thr_mma.make_fragment_A(sA)
        frag_B = thr_mma.make_fragment_B(sB)
        frag_C = thr_mma.make_fragment_C(gC)
        frag_A_retile = thr_copy_A.retile(frag_A)
        frag_B_retile = thr_copy_B.retile(frag_B)
        frag_C.fill(0.0)
        r2g_atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_elem)
        thr_copy_C = fx.make_tiled_copy_C(r2g_atom, tiled_mma).get_slice(tid)
        thr_gC = thr_copy_C.partition_S(gC)

        a_row_coords = fx.make_view(0, fx.make_layout((block_m, block_k), (1, 0)))
        b_row_coords = fx.make_view(0, fx.make_layout((block_n, block_k), (1, 0)))
        thr_mma_aRow = thr_mma.partition_A(a_row_coords)
        thr_mma_bRow = thr_mma.partition_B(b_row_coords)
        # The scaled-MFMA state selects one E8M0 byte for each 32-K lane group.
        scale_group = (fx.Int32(tid) % fx.Int32(GFX950_WAVE_SIZE)) // fx.Int32(
            MXFP8_MFMA_N
        )

        wave_offset = rocdl.readfirstlane(
            fx.Int64.ir_type,
            fx.Int64(
                fx.Int32(tid)
                // fx.Int32(GFX950_WAVE_SIZE)
                * fx.Int32(GFX950_WAVE_SIZE * GFX950_DMA_BYTES)
            ),
        )

        def make_wave_lds_ptr(ptr):
            return ptr + fx.Int32(wave_offset)

        def swizzled_col(row, col, layout):
            return fx.get_scalar(fx.crd2idx((row, col), layout)) % fx.Int32(block_k)

        def async_load_tile(
            gmem,
            smem,
            stage_bytes,
            ldg_iters,
            rows_base,
            k_tile,
            stage,
            layout,
        ):
            # Direct-to-LDS stores are linear, so the source coordinate carries
            # the composed LDS swizzle.
            lds_ptr = make_wave_lds_ptr(smem + stage * fx.Int32(stage_bytes))
            for i in range_constexpr(ldg_iters):
                lin = (fx.Int32(i * block_threads) + fx.Int32(tid)) * fx.Int32(
                    GFX950_DMA_BYTES
                )
                row = lin // fx.Int32(block_k)
                dst_col = lin % fx.Int32(block_k)
                src_col = swizzled_col(row, dst_col, layout)
                src_offset = (
                    (rows_base + row) * fx.Int32(k)
                    + k_tile * fx.Int32(block_k)
                    + src_col
                )
                src = fx.slice(gmem, (None, src_offset))
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                fx.copy(dma_atom, src, dst)
                if i < ldg_iters - 1:
                    lds_ptr = lds_ptr + fx.Int32(block_threads * GFX950_DMA_BYTES)

        def async_load_a(k_tile, stage):
            async_load_tile(
                a_flat,
                smem_a,
                d.a_stage_bytes,
                d.ldg_a_iters,
                m_base,
                k_tile,
                stage,
                a_lds_layout,
            )

        def async_load_b(k_tile, stage):
            async_load_tile(
                b_flat,
                smem_b,
                d.b_stage_bytes,
                d.ldg_b_iters,
                n_base,
                k_tile,
                stage,
                b_lds_layout,
            )

        def load_scale_word(scale, row_global, scale_col):
            scale_offset = fx.Int32(row_global) * fx.Int32(scale_k) + fx.Int32(
                scale_col
            )
            scale_reg = fx.make_rmem_tensor(1, fx.Uint8)
            fx.copy(
                scale_atom,
                fx.slice(scale, (None, scale_offset)),
                scale_reg,
            )
            scale_byte = fx.get_scalar(scale_reg[0])
            return scale_byte.to(fx.Int32) * fx.Int32(0x01010101)

        def scaled_mma(d_frag, a_frag, b_frag, scale_a, scale_b):
            # Ownership and fragments come from the workgroup TiledMma, while
            # dynamic E8M0 state is bound on each underlying 16x16 atom call.
            a_atom = fx.Tensor(
                fx.make_view(fx.get_iter(a_frag), fx.coalesce(a_frag.layout))
            )
            b_atom = fx.Tensor(
                fx.make_view(fx.get_iter(b_frag), fx.coalesce(b_frag.layout))
            )
            fx.gemm(
                mma_atom,
                d_frag,
                a_atom,
                b_atom,
                d_frag,
                scale_a=scale_a,
                scale_b=scale_b,
            )

        # Packed scale path. One 4-byte load holds a whole K-tile of E8M0 scales
        # for one row, and 64 lanes cover four MMA repeats at once, so
        # ds_bpermute can do the 4x4 lane-group transpose that hands each row's
        # dword to the lane group that needs it; each lane then shifts out its
        # own K-quarter byte. This turns 16 single-byte loads per K-tile into 4
        # dword loads, and it is the number of outstanding scale loads, not
        # their bytes, that the MFMA stream waits on.
        scale32_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Uint32)
        scale_k32 = scale_k // 4
        lane_row = (fx.Int32(tid) % fx.Int32(GFX950_WAVE_SIZE)) % fx.Int32(MXFP8_MFMA_N)

        def make_flat_buffer32(tensor, elems32):
            # The scale tensors arrive as u8 views, so their pointer carries
            # alignment 1 and a 4-byte load needs that restated.
            src = fx.get_iter(tensor)
            flat = fx.Tensor(
                fx.make_view(
                    fx.recast_iter(
                        fx.PointerType.get(fx.Uint32.ir_type, src.memspace, 4), src
                    ),
                    fx.make_layout(elems32, 1),
                )
            )
            return fx.rocdl.make_buffer_tensor(flat, max_size=True)

        if const_expr(packed_scale):
            sa32 = fx.logical_divide(
                make_flat_buffer32(scale_a_u8, m * scale_k32), fx.make_layout(1, 1)
            )
            sb32 = fx.logical_divide(
                make_flat_buffer32(scale_b_u8, n * scale_k32), fx.make_layout(1, 1)
            )

        def packed_scale_words(buf, base, thr_row, n_repeat, kh, col32):
            rows = [fx.get_scalar(thr_row[0, mi, kh]) for mi in range(n_repeat)]
            repeat_stride = rows[1] - rows[0]
            words = []
            for q in range_constexpr(0, n_repeat, 4):
                row = rows[q] + repeat_stride * scale_group
                offset = (base + row) * fx.Int32(scale_k32) + col32
                reg = fx.make_rmem_tensor(1, fx.Uint32)
                fx.copy(scale32_atom, fx.slice(buf, (None, offset)), reg)
                packed = fx.get_scalar(reg[0]).to(fx.Int32)
                for j in range_constexpr(4):
                    idx = (fx.Int32(j * MXFP8_MFMA_N) + lane_row) * fx.Int32(4)
                    lane_word = fx.Int32(
                        rocdl.ds_bpermute(fx.Int32.ir_type, idx, packed)
                    )
                    byte = (lane_word >> (scale_group * fx.Int32(8))) & fx.Int32(0xFF)
                    words.append(byte * fx.Int32(0x01010101))
            return words

        def load_fragments(stage):
            sA_stage = fx.make_view(
                smem_a + stage * fx.Int32(block_m * block_k),
                a_lds_layout,
            )
            sB_stage = fx.make_view(
                smem_b + stage * fx.Int32(block_n * block_k),
                b_lds_layout,
            )
            thr_sA = thr_copy_A.partition_S(sA_stage)
            thr_sB = thr_copy_B.partition_S(sB_stage)
            for kh in range_constexpr(d.k_halves):
                fx.copy(
                    s2r_atom,
                    thr_sB[None, None, kh],
                    frag_B_retile[None, None, kh],
                )
                fx.copy(
                    s2r_atom,
                    thr_sA[None, None, kh],
                    frag_A_retile[None, None, kh],
                )

        def mma_stage(k_tile):
            for kh in range_constexpr(d.k_halves):
                if const_expr(packed_scale):
                    col32 = k_tile * fx.Int32(block_k // MXFP8_MFMA_K) + fx.Int32(kh)
                    sa_words = packed_scale_words(
                        sa32, m_base, thr_mma_aRow, d.mma_m_repeat, kh, col32
                    )
                    sb_words = packed_scale_words(
                        sb32, n_base, thr_mma_bRow, d.mma_n_repeat, kh, col32
                    )
                else:
                    scale_col = (
                        k_tile * fx.Int32(block_k // MXFP8_SCALE_BLOCK_K)
                        + fx.Int32(kh * (MXFP8_MFMA_K // MXFP8_SCALE_BLOCK_K))
                        + scale_group
                    )
                    sa_words = []
                    for mi in range_constexpr(d.mma_m_repeat):
                        local_row = fx.get_scalar(thr_mma_aRow[0, mi, kh])
                        sa_words.append(
                            load_scale_word(
                                sa_flat,
                                m_base + local_row,
                                scale_col,
                            )
                        )

                    sb_words = []
                    for ni in range_constexpr(d.mma_n_repeat):
                        local_row = fx.get_scalar(thr_mma_bRow[0, ni, kh])
                        sb_words.append(
                            load_scale_word(
                                sb_flat,
                                n_base + local_row,
                                scale_col,
                            )
                        )

                for ni in range_constexpr(d.mma_n_repeat):
                    for mi in range_constexpr(d.mma_m_repeat):
                        scaled_mma(
                            frag_C[(None, 0), mi, ni],
                            frag_A[None, mi, kh],
                            frag_B[None, ni, kh],
                            sa_words[mi],
                            sb_words[ni],
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
            # A direct-to-LDS load is a VMEM op that writes LDS, so the compiler
            # cannot prove it does not alias a later ds_read and drains vmcnt to
            # zero in between. With frag_first the tile's fragments are read
            # first, which leaves the DMA free to overlap the whole MFMA block;
            # workgroups small enough to keep several resident per CU hide that
            # latency across waves instead and prefer issuing the DMA earliest.
            if const_expr(frag_first):
                load_fragments(cur)
                async_load_b(k_tile + fx.Int32(stages - 1), write)
                async_load_a(k_tile + fx.Int32(stages - 1), write)
            else:
                async_load_b(k_tile + fx.Int32(stages - 1), write)
                async_load_a(k_tile + fx.Int32(stages - 1), write)
                load_fragments(cur)
            mma_stage(k_tile)

        # Drain: consume the buffers still in flight, walking vmcnt down to 0.
        for s in range_constexpr(stages - 1):
            k_tile = fx.Int32(main_loop_end + s)
            cur = k_tile % fx.Int32(stages)
            __barrier((stages - 2 - s) * d.ldg_wait_count)
            load_fragments(cur)
            mma_stage(k_tile)

        frag_C_out = fx.make_fragment_like(frag_C, out_elem)
        frag_C_out.store(frag_C.load().to(out_elem))
        frag_C_retile = thr_copy_C.retile(frag_C_out)
        fx.copy(r2g_atom, frag_C_retile, thr_gC)

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
            frag_first=frag_first,
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
