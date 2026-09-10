# SPDX-License-Identifier: BSD-3-Clause

"""gfx950 MXFP4 and MXFP8 scaled GEMM.

Both formats use per-32-element E8M0 block scales and CDNA4 16x16x128 scaled
MFMA instructions. This kernel shares tiling, direct-to-LDS DMA, scale loading,
the staged waitcnt pipeline, and the output epilogue between the two formats. A
compile-time format policy selects the operand type and fragment layout. MXFP8
stores one E4M3 value per byte; MXFP4 stores two E2M1 values per byte, so
logical K and storage K remain distinct throughout the address calculations.
"""

from dataclasses import dataclass
from typing import Literal

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from .gemm_gfx950 import (
    __barrier,
    AsyncLoadContext,
    AsyncLoadOperand,
    BlockSwizzle,
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
    async_load_operand,
    get_wave_lds_offset,
)


def _make_wave_layout(m_waves, n_waves):
    return fx.make_layout((m_waves, n_waves, 1), (n_waves, 1, 0))


def _make_row_major_lds_layout(rows, inner_extent, swizzle):
    return fx.make_composed_layout(
        swizzle,
        fx.make_ordered_layout((rows, inner_extent), (1, 0)),
    )


def _make_transposed_lds_layout(rows, inner_extent, granule_bits):
    base_layout = fx.make_ordered_layout((rows, inner_extent), (0, 1))
    if const_expr(rows == 64):
        return fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(2, granule_bits, 2)), base_layout
        )
    if const_expr(rows == 128):
        return fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(2, granule_bits, 3)), base_layout
        )
    if const_expr(rows == 256):
        return fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(2, granule_bits, 4)), base_layout
        )
    return base_layout


def _permlane_swap(width, old, src):
    """v_permlane{16,32}_swap_b32 -> (new_old, new_src) as i32 IR values.

    Both operands are read-modify-write. The selected instruction exchanges
    register values between 16- or 32-lane partitions and returns both results.
    """
    i32 = ir.IntegerType.get_signless(32)
    sty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    fn = _rocdl_ops.permlane16_swap if width == 16 else _rocdl_ops.permlane32_swap
    res = fn(sty, fx.as_ir_value(old), fx.as_ir_value(src), False, False)
    return llvm.extractvalue(i32, res, [0]), llvm.extractvalue(i32, res, [1])


def _ds_read_tr8_b64(addr_i32):
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    return llvm.inline_asm(
        raw_type,
        [fx.as_ir_value(addr_i32)],
        "ds_read_b64_tr_b8 $0, $1 offset:0\n",
        "=v,v,~{memory}",
        has_side_effects=True,
    )


# Shared MXFP format and gfx950 hardware constants.
MXFPFormat = Literal["mxfp4", "mxfp8"]
MXFP_SCALE_BLOCK_K = 32
MXFP_MFMA_M = 16
MXFP_MFMA_N = 16
MXFP_MFMA_K = 128
GFX950_SCALE_DMA_BYTES = 4
GFX950_LDS_CAPACITY = 163840
GFX950_MAX_BLOCK_THREADS = 1024
MXFP_OUT_BYTES = 2
MXFP_MAX_MMA_REPEAT = 8
MXFP_FORMAT_FP4 = 4
MXFP_FORMAT_FP8 = 8
MXFP_OUT_DTYPE_BF16 = 2
MXFP_OUT_DTYPE_FP16 = 3


def _elements_per_byte(mxfp_format: MXFPFormat) -> int:
    if mxfp_format == "mxfp8":
        return 1
    if mxfp_format == "mxfp4":
        return 2
    raise ValueError(f"unsupported MXFP operand format: {mxfp_format}")


@fx.struct
class MXFPGemmParams:
    mxfp_format_id: fx.Constexpr[int]
    m: fx.Constexpr[int]
    n: fx.Constexpr[int]
    k: fx.Constexpr[int]
    out_dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    stages: fx.Constexpr[int]
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    group_m: fx.Constexpr[int]
    lds_scale: fx.Constexpr[bool]
    a_is_transposed: fx.Constexpr[bool]
    b_is_transposed: fx.Constexpr[bool]
    use_cshuffle: fx.Constexpr[bool]
    block_threads: fx.Constexpr[int]
    block_k_bytes: fx.Constexpr[int]
    mma_m_repeat: fx.Constexpr[int]
    mma_n_repeat: fx.Constexpr[int]
    k_halves: fx.Constexpr[int]
    granules_per_row: fx.Constexpr[int]
    ldg_a_iters: fx.Constexpr[int]
    ldg_b_iters: fx.Constexpr[int]
    ldg_wait_count: fx.Constexpr[int]
    sc_a_iters: fx.Constexpr[int]
    sc_b_iters: fx.Constexpr[int]
    sc_a_bytes: fx.Constexpr[int]
    sc_b_bytes: fx.Constexpr[int]
    scale_row_bytes: fx.Constexpr[int]
    a_stage_bytes: fx.Constexpr[int]
    b_stage_bytes: fx.Constexpr[int]


@dataclass(frozen=True)
class MXFPGemmDerived:
    """Tile quantities shared by the kernel and heuristics validator."""

    block_threads: int
    block_k_bytes: int
    mma_m_repeat: int
    mma_n_repeat: int
    k_halves: int
    granules_per_row: int
    ldg_a_iters: int
    ldg_b_iters: int
    ldg_wait_count: int
    lds_scale: bool
    sc_a_iters: int
    sc_b_iters: int
    sc_a_bytes: int
    sc_b_bytes: int
    scale_row_bytes: int
    a_stage_bytes: int
    b_stage_bytes: int
    use_cshuffle: bool


def mxfp_gemm_derived(
    mxfp_format: MXFPFormat,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int = 0,
    lds_scale_req: int = 0,
) -> MXFPGemmDerived:
    """Validate a tile config and return its derived quantities.

    block_m, block_n, and block_k are all in elements. Raises ValueError for
    any config the kernel cannot express.
    """
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise ValueError("block_m, block_n, and block_k must be positive")
    if stages < 2:
        raise ValueError("stages must be at least 2 for the staged LDS pipeline")
    if m_waves <= 0 or n_waves <= 0:
        raise ValueError("m_waves and n_waves must be positive")
    if group_m < 0:
        raise ValueError("group_m must be non-negative")
    elements_per_byte = _elements_per_byte(mxfp_format)
    if block_k % MXFP_MFMA_K != 0:
        raise ValueError(
            f"block_k must be a multiple of the MFMA K depth: block_k={block_k}"
        )
    block_k_bytes = block_k // elements_per_byte

    block_threads = m_waves * n_waves * GFX950_WAVE_SIZE
    if block_threads > GFX950_MAX_BLOCK_THREADS:
        raise ValueError(f"block exceeds {GFX950_MAX_BLOCK_THREADS} threads")

    wave_tile_m, rem_m = divmod(block_m, m_waves)
    wave_tile_n, rem_n = divmod(block_n, n_waves)
    if rem_m or rem_n:
        raise ValueError("block_m/block_n must be divisible by m_waves/n_waves")

    mma_m_repeat, rem_m = divmod(wave_tile_m, MXFP_MFMA_M)
    mma_n_repeat, rem_n = divmod(wave_tile_n, MXFP_MFMA_N)
    if rem_m or rem_n or mma_m_repeat == 0 or mma_n_repeat == 0:
        raise ValueError(
            "each wave tile must be a positive multiple of the 16x16 MFMA tile"
        )
    if mma_m_repeat > MXFP_MAX_MMA_REPEAT or mma_n_repeat > MXFP_MAX_MMA_REPEAT:
        raise ValueError(
            "accumulator repeats exceed the register budget: "
            f"mma_m_repeat={mma_m_repeat}, mma_n_repeat={mma_n_repeat}"
        )

    cshuffle_vec_size = GFX950_DMA_BYTES // MXFP_OUT_BYTES
    cshuffle_x_threads = block_n // cshuffle_vec_size
    use_cshuffle = (
        block_n % cshuffle_vec_size == 0
        and block_threads % cshuffle_x_threads == 0
        and block_m % (block_threads // cshuffle_x_threads) == 0
    )

    granules_per_row = block_k_bytes // GFX950_DMA_BYTES
    if granules_per_row == 0 or granules_per_row & (granules_per_row - 1):
        raise ValueError(
            "the packed K-row byte count divided by the DMA width must be a "
            f"power of two for the XOR swizzle: block_k={block_k}, "
            f"block_k_bytes={block_k_bytes}"
        )

    dma_bytes_per_pass = block_threads * GFX950_DMA_BYTES
    if (block_m * block_k_bytes) % dma_bytes_per_pass != 0:
        raise ValueError(
            "A tile load schedule must exactly cover the LDS tile: "
            f"block_m={block_m}, block_k={block_k}, block_threads={block_threads}"
        )
    if (block_n * block_k_bytes) % dma_bytes_per_pass != 0:
        raise ValueError(
            "B tile load schedule must exactly cover the LDS tile: "
            f"block_n={block_n}, block_k={block_k}, block_threads={block_threads}"
        )
    ldg_a_iters = (block_m * block_k_bytes) // dma_bytes_per_pass
    ldg_b_iters = (block_n * block_k_bytes) // dma_bytes_per_pass

    a_stage_bytes = block_m * block_k_bytes
    b_stage_bytes = block_n * block_k_bytes

    # ---- LDS-staged E8M0 scales -------------------------------------------
    scale_row_bytes = block_k // MXFP_SCALE_BLOCK_K
    sc_bytes_per_pass = block_threads * GFX950_SCALE_DMA_BYTES
    sc_a_bytes = block_m * scale_row_bytes
    sc_b_bytes = block_n * scale_row_bytes
    scale_dma_exact = (
        sc_a_bytes % sc_bytes_per_pass == 0 and sc_b_bytes % sc_bytes_per_pass == 0
    )
    tile_bytes = stages * (a_stage_bytes + b_stage_bytes)
    if tile_bytes > GFX950_LDS_CAPACITY:
        raise ValueError(
            "staged LDS buffers exceed the device shared-memory capacity: "
            f"stages={stages}, block_m={block_m}, "
            f"block_n={block_n}, block_k={block_k}, smem_bytes={tile_bytes}, "
            f"capacity={GFX950_LDS_CAPACITY}"
        )

    lds_scale = scale_dma_exact and bool(lds_scale_req)
    if lds_scale_req and not scale_dma_exact:
        raise ValueError(
            "LDS-staged scales need each scale tile to cover a whole DMA pass: "
            f"block_m={block_m}, block_n={block_n}, block_k={block_k}, "
            f"block_threads={block_threads}"
        )
    sc_a_iters = sc_a_bytes // sc_bytes_per_pass if lds_scale else 0
    sc_b_iters = sc_b_bytes // sc_bytes_per_pass if lds_scale else 0
    scale_bytes = stages * (sc_a_bytes + sc_b_bytes) if lds_scale else 0
    ab_scale_bytes = tile_bytes + scale_bytes
    cshuffle_bytes = block_m * block_n * MXFP_OUT_BYTES if use_cshuffle else 0
    if max(ab_scale_bytes, cshuffle_bytes) > GFX950_LDS_CAPACITY:
        raise ValueError(
            "staged tiles and C-shuffle exceed the device shared-memory capacity: "
            f"ab_scale_bytes={ab_scale_bytes}, cshuffle_bytes={cshuffle_bytes}, "
            f"capacity={GFX950_LDS_CAPACITY}"
        )

    ldg_wait_count = ldg_a_iters + sc_a_iters + ldg_b_iters + sc_b_iters
    # Scale DMA contributes to the vmcnt wait budget.
    if (stages - 2) * ldg_wait_count >= 63:
        if lds_scale:
            raise ValueError(
                "the scale DMA lengthens the in-order vmcnt chain past the wait "
                "budget for this pipeline depth"
            )
        raise ValueError("staged pipeline wait count exceeds supported range")

    return MXFPGemmDerived(
        block_threads=block_threads,
        block_k_bytes=block_k_bytes,
        mma_m_repeat=mma_m_repeat,
        mma_n_repeat=mma_n_repeat,
        k_halves=block_k // MXFP_MFMA_K,
        granules_per_row=granules_per_row,
        ldg_a_iters=ldg_a_iters,
        ldg_b_iters=ldg_b_iters,
        ldg_wait_count=ldg_wait_count,
        lds_scale=lds_scale,
        sc_a_iters=sc_a_iters,
        sc_b_iters=sc_b_iters,
        sc_a_bytes=sc_a_bytes,
        sc_b_bytes=sc_b_bytes,
        scale_row_bytes=scale_row_bytes,
        a_stage_bytes=a_stage_bytes,
        b_stage_bytes=b_stage_bytes,
        use_cshuffle=use_cshuffle,
    )


def make_mxfp_param_and_validate(
    mxfp_format: MXFPFormat,
    m,
    n,
    k,
    out_dtype,
    gemm_config,
    a_is_transposed=False,
    b_is_transposed=True,
):
    """Return one concrete MXFP specialization, or None if unsupported."""
    if m <= 0 or n <= 0 or k <= 0:
        return None
    if out_dtype not in ("bfloat16", "float16"):
        return None
    block_m = int(gemm_config["TILE_M"])
    block_n = int(gemm_config["TILE_N"])
    block_k = int(gemm_config["TILE_K"])
    stages = int(gemm_config["STAGES"])
    m_waves = int(gemm_config["M_WAVES"])
    n_waves = int(gemm_config["N_WAVES"])
    group_m = int(gemm_config["GROUP_M"])
    lds_scale = int(gemm_config.get("LDS_SCALE", 0))
    try:
        derived = mxfp_gemm_derived(
            mxfp_format,
            block_m,
            block_n,
            block_k,
            stages,
            m_waves,
            n_waves,
            group_m,
            lds_scale_req=lds_scale,
        )
    except ValueError:
        return None

    if m % block_m or n % block_n or k % block_k:
        return None
    if (k // block_k) <= stages - 1:
        return None
    return MXFPGemmParams(
        mxfp_format_id=MXFP_FORMAT_FP4 if mxfp_format == "mxfp4" else MXFP_FORMAT_FP8,
        m=m,
        n=n,
        k=k,
        out_dtype_id=(
            MXFP_OUT_DTYPE_BF16 if out_dtype == "bfloat16" else MXFP_OUT_DTYPE_FP16
        ),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
        lds_scale=derived.lds_scale,
        a_is_transposed=a_is_transposed,
        b_is_transposed=b_is_transposed,
        use_cshuffle=derived.use_cshuffle,
        block_threads=derived.block_threads,
        block_k_bytes=derived.block_k_bytes,
        mma_m_repeat=derived.mma_m_repeat,
        mma_n_repeat=derived.mma_n_repeat,
        k_halves=derived.k_halves,
        granules_per_row=derived.granules_per_row,
        ldg_a_iters=derived.ldg_a_iters,
        ldg_b_iters=derived.ldg_b_iters,
        ldg_wait_count=derived.ldg_wait_count,
        sc_a_iters=derived.sc_a_iters,
        sc_b_iters=derived.sc_b_iters,
        sc_a_bytes=derived.sc_a_bytes,
        sc_b_bytes=derived.sc_b_bytes,
        scale_row_bytes=derived.scale_row_bytes,
        a_stage_bytes=derived.a_stage_bytes,
        b_stage_bytes=derived.b_stage_bytes,
    )


def make_mxfp_gemm_kernel_name(param: MXFPGemmParams) -> str:
    mxfp_format = "mxfp4" if param.mxfp_format_id == MXFP_FORMAT_FP4 else "mxfp8"
    out_dtype = (
        "bfloat16" if param.out_dtype_id == MXFP_OUT_DTYPE_BF16 else "float16"
    )
    return (
        f"{mxfp_format}_scaled_mm_gfx950"
        f"_{out_dtype}"
        f"_bm{param.block_m}_bn{param.block_n}_bk{param.block_k}"
        f"_s{param.stages}_mw{param.m_waves}_nw{param.n_waves}"
        f"_g{param.group_m}"
        f"_ls{param.lds_scale}"
        f"_l{'t' if param.a_is_transposed else 'n'}"
        f"{'t' if param.b_is_transposed else 'n'}"
    )


def make_mxfp_tiled_mma(param: MXFPGemmParams, operand_elem):
    mma_atom = fx.make_mma_atom(
        fx.rocdl.cdna4.MFMA_Scale(
            MXFP_MFMA_M,
            MXFP_MFMA_N,
            MXFP_MFMA_K,
            operand_elem,
            operand_elem,
            fx.Float32,
            opsel_a=0,
            opsel_b=0,
        )
    )
    wave_layout = _make_wave_layout(param.m_waves, param.n_waves)
    if const_expr(param.mxfp_format_id == MXFP_FORMAT_FP4):
        return mma_atom, fx.make_tiled_mma(mma_atom, wave_layout)
    mma_permutation = fx.make_tile(
        None,
        None,
        fx.make_layout(
            (GFX950_DMA_BYTES, 2, MXFP_MFMA_K // (2 * GFX950_DMA_BYTES)),
            (1, MXFP_MFMA_K // 2, GFX950_DMA_BYTES),
        ),
    )
    return mma_atom, fx.make_tiled_mma(mma_atom, wave_layout, mma_permutation)


def make_mxfp_ab_lds_layouts(
    block_m, block_n, block_k_bytes, a_is_transposed, b_is_transposed
):
    granules_per_row = block_k_bytes // GFX950_DMA_BYTES
    swizzle_bits = granules_per_row.bit_length() - 1
    granule_byte_bits = GFX950_DMA_BYTES.bit_length() - 1
    swizzle_bytes = fx.static(
        fx.SwizzleType.get(swizzle_bits, granule_byte_bits, swizzle_bits)
    )
    return (
        _make_transposed_lds_layout(block_m, block_k_bytes, 4)
        if a_is_transposed
        else _make_row_major_lds_layout(block_m, block_k_bytes, swizzle_bytes),
        _make_transposed_lds_layout(block_n, block_k_bytes, 4)
        if not b_is_transposed
        else _make_row_major_lds_layout(block_n, block_k_bytes, swizzle_bytes),
    )


@flyc.kernel
def gemm_mxfp_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b_nk: fx.Tensor,
    scale_a_u8: fx.Tensor,
    scale_b_u8: fx.Tensor,
    param: MXFPGemmParams,
):
    d = param
    m = param.m
    n = param.n
    k = param.k
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    stages = param.stages
    m_waves = param.m_waves
    n_waves = param.n_waves
    group_m = param.group_m
    a_is_transposed = param.a_is_transposed
    b_is_transposed = param.b_is_transposed
    is_mxfp4 = param.mxfp_format_id == MXFP_FORMAT_FP4
    elements_per_byte = 2 if const_expr(is_mxfp4) else 1
    operand_elem = fx.Float4E2M1FN if const_expr(is_mxfp4) else fx.Float8E4M3FN
    out_elem = (
        fx.BFloat16
        if const_expr(param.out_dtype_id == MXFP_OUT_DTYPE_BF16)
        else fx.Float16
    )
    block_threads = param.block_threads
    block_k_bytes = param.block_k_bytes
    granules_per_row = param.granules_per_row
    k_bytes = k // elements_per_byte
    scale_k = k // MXFP_SCALE_BLOCK_K
    tiles_m = m // block_m
    tiles_n = n // block_n
    packed_repeat_scale = (
        param.mma_m_repeat % 4 == 0 and param.mma_n_repeat % 4 == 0
    )
    a_scale_units = param.mma_m_repeat * param.k_halves
    b_scale_units = param.mma_n_repeat * param.k_halves
    packed_unit_scale = not packed_repeat_scale and (
        -(-a_scale_units // 4) + -(-b_scale_units // 4)
        < param.k_halves * (param.mma_m_repeat + param.mma_n_repeat)
    )
    packed_scale = packed_repeat_scale or packed_unit_scale

    tid = fx.thread_idx.x

    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=group_m
    )
    bid_m, bid_n = block_swizzle.swizzle(tiles_m, tiles_n, fx.block_idx.x)
    block_m_offset = bid_m * fx.Int32(block_m)
    block_n_offset = bid_n * fx.Int32(block_n)

    if const_expr(d.lds_scale):

        @fx.struct
        class SharedABStorage:
            a: fx.Array[operand_elem, stages * block_m * block_k, 16]
            b: fx.Array[operand_elem, stages * block_n * block_k, 16]
            # E8M0 bytes, one per 32 elements, staged on the same schedule.
            sca: fx.Array[fx.Uint8, stages * d.sc_a_bytes, 16]
            scb: fx.Array[fx.Uint8, stages * d.sc_b_bytes, 16]

    else:

        @fx.struct
        class SharedABStorage:
            a: fx.Array[operand_elem, stages * block_m * block_k, 16]
            b: fx.Array[operand_elem, stages * block_n * block_k, 16]

    if const_expr(param.use_cshuffle):

        @fx.union
        class SharedStorage:
            ab: SharedABStorage
            c: fx.Array[out_elem, block_m * block_n, 16]

        storage = fx.SharedAllocator().allocate(SharedStorage)
        ab_storage = storage.ab.peek()
        smem_c = storage.c.peek().ptr
    else:
        ab_storage = fx.SharedAllocator().allocate(SharedABStorage).peek()

    smem_a = ab_storage.a.ptr
    smem_b = ab_storage.b.ptr

    smem_a_bytes = fx.recast_iter(fx.Uint8, ab_storage.a.ptr)
    smem_b_bytes = fx.recast_iter(fx.Uint8, ab_storage.b.ptr)
    if const_expr(d.lds_scale):
        smem_sca = fx.recast_iter(fx.Uint8, ab_storage.sca.ptr)
        smem_scb = fx.recast_iter(fx.Uint8, ab_storage.scb.ptr)

    def make_flat_buffer(tensor, elems):
        flat = fx.Tensor(
            fx.make_view(fx.get_iter(tensor), fx.make_layout(elems, 1))
        )
        return fx.rocdl.make_buffer_tensor(flat, max_size=True)

    a_leading_stride = fx.Int32(
        fx.get_scalar(a.stride[1] if a_is_transposed else a.stride[0])
    )
    b_leading_stride = fx.Int32(
        fx.get_scalar(b_nk.stride[1] if not b_is_transposed else b_nk.stride[0])
    )

    # A and B arrive as uint8 views, so their flat extents are byte counts.
    a_flat = make_flat_buffer(a, m * k_bytes)
    b_flat = make_flat_buffer(b_nk, n * k_bytes)
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

    mma_atom, tiled_mma = make_mxfp_tiled_mma(param, operand_elem)
    thr_mma = tiled_mma.thr_slice(tid)

    lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
    if const_expr(not is_mxfp4):
        universal_s2r_atom = fx.make_copy_atom(
            fx.UniversalCopy128b(), operand_elem
        )
        buffer_s2r_atom = fx.make_copy_atom(
            fx.rocdl.BufferCopy128b(), operand_elem
        )
        transposed_s2r_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans8_64b(), operand_elem
        )
        a_s2r_atom = (
            transposed_s2r_atom if a_is_transposed else universal_s2r_atom
        )
        b_s2r_atom = (
            transposed_s2r_atom if not b_is_transposed else universal_s2r_atom
        )
        a_tiled_copy_atom = (
            transposed_s2r_atom if a_is_transposed else buffer_s2r_atom
        )
        b_tiled_copy_atom = (
            transposed_s2r_atom if not b_is_transposed else buffer_s2r_atom
        )
        thr_copy_A = fx.make_tiled_copy_A(a_tiled_copy_atom, tiled_mma).get_slice(
            tid
        )
        thr_copy_B = fx.make_tiled_copy_B(b_tiled_copy_atom, tiled_mma).get_slice(
            tid
        )
    scale_atom = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), fx.Uint8)

    a_lds_layout_bytes, b_lds_layout_bytes = make_mxfp_ab_lds_layouts(
        block_m,
        block_n,
        block_k_bytes,
        a_is_transposed,
        b_is_transposed,
    )

    if const_expr(not is_mxfp4):
        sA = fx.make_view(smem_a, a_lds_layout_bytes)
        sB = fx.make_view(smem_b, b_lds_layout_bytes)
        frag_A = thr_mma.make_fragment_A(sA)
        frag_B = thr_mma.make_fragment_B(sB)
        frag_A_retile = thr_copy_A.retile(frag_A)
        frag_B_retile = thr_copy_B.retile(frag_B)

    frag_C = thr_mma.make_fragment_C(gC)
    frag_C.fill(0.0)

    if const_expr(param.use_cshuffle):
        c_lds_layout = fx.make_layout((block_m, block_n), (block_n, 1))
        sC = fx.make_view(smem_c, c_lds_layout)
        row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
        col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
        thr_mma_cRow = thr_mma.partition_C(row_coords)
        thr_mma_cCol = thr_mma.partition_C(col_coords)
        cshuffle_s2r_atom = fx.make_copy_atom(fx.UniversalCopy128b(), out_elem)
        cshuffle_r2g_atom = fx.make_copy_atom(
            fx.rocdl.BufferCopy128b(), out_elem
        )
        cshuffle_vec_size = GFX950_DMA_BYTES // MXFP_OUT_BYTES
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
            cshuffle_r2g_atom,
            cshuffle_tv_layout,
            cshuffle_tile,
        )
        thr_copy_cshuffle = tiled_copy_cshuffle.get_slice(tid)
        thr_sC = thr_copy_cshuffle.partition_S(sC)
        thr_gC = thr_copy_cshuffle.partition_D(gC)
        frag_C_cshuffle = fx.make_fragment_like(thr_sC)
    else:
        r2g_atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_elem)
        thr_copy_C = fx.make_tiled_copy_C(r2g_atom, tiled_mma).get_slice(tid)
        thr_gC = thr_copy_C.partition_S(gC)

    lane = fx.Int32(tid) % fx.Int32(GFX950_WAVE_SIZE)
    wave = rocdl.readfirstlane(
        fx.Int32.ir_type, fx.Int32(tid) // fx.Int32(GFX950_WAVE_SIZE)
    )
    wave_m = fx.Int32(wave) // fx.Int32(n_waves)
    wave_n = fx.Int32(wave) % fx.Int32(n_waves)
    lane_row = lane % fx.Int32(MXFP_MFMA_M)
    lane_grp = lane // fx.Int32(MXFP_MFMA_M)
    m_repeat_stride = m_waves * MXFP_MFMA_M
    n_repeat_stride = n_waves * MXFP_MFMA_N
    a_row_base = wave_m * fx.Int32(MXFP_MFMA_M) + lane_row
    b_row_base = wave_n * fx.Int32(MXFP_MFMA_N) + lane_row
    # 16-byte granules spanned by one 128-element MFMA K step.
    granules_per_kh = MXFP_MFMA_K // (elements_per_byte * GFX950_DMA_BYTES)
    row_dwords = block_k_bytes // 4

    ab_load_context = AsyncLoadContext(
        wave_offset=get_wave_lds_offset(tid, GFX950_DMA_BYTES),
        tid=tid,
        inner_bound=k_bytes,
        block_threads=block_threads,
        async_load_bytes=GFX950_DMA_BYTES,
        in_data_bytes=1,
        ldg_x_threads=block_k_bytes // GFX950_DMA_BYTES,
        block_k=block_k_bytes,
        has_k_tail=False,
    )
    a_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        rsrc=fx.rocdl.get_buffer_rsrc(fx.get_iter(a_flat)),
        lds_layout=a_lds_layout_bytes,
        outer_tile_size=block_m,
        outer_bound=m,
        leading_stride=a_leading_stride,
        load_iters=d.ldg_a_iters,
        is_k_major=a_is_transposed,
        has_outer_tail=False,
    )
    b_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        rsrc=fx.rocdl.get_buffer_rsrc(fx.get_iter(b_flat)),
        lds_layout=b_lds_layout_bytes,
        outer_tile_size=block_n,
        outer_bound=n,
        leading_stride=b_leading_stride,
        load_iters=d.ldg_b_iters,
        is_k_major=not b_is_transposed,
        has_outer_tail=False,
    )

    if const_expr(d.lds_scale):
        sc_lds_atom = fx.make_copy_atom(fx.UniversalCopy8b(), fx.Uint8)
        scale_load_context = AsyncLoadContext(
            wave_offset=get_wave_lds_offset(tid, GFX950_SCALE_DMA_BYTES),
            tid=tid,
            inner_bound=scale_k,
            block_threads=block_threads,
            async_load_bytes=GFX950_SCALE_DMA_BYTES,
            in_data_bytes=1,
            ldg_x_threads=d.scale_row_bytes // GFX950_SCALE_DMA_BYTES,
            block_k=d.scale_row_bytes,
            has_k_tail=False,
        )
        a_scale_load_operand = AsyncLoadOperand(
            context=scale_load_context,
            rsrc=fx.rocdl.get_buffer_rsrc(fx.get_iter(sa_flat)),
            lds_layout=fx.make_ordered_layout((block_m, d.scale_row_bytes), (1, 0)),
            outer_tile_size=block_m,
            outer_bound=m,
            leading_stride=scale_k,
            load_iters=d.sc_a_iters,
            is_k_major=False,
            has_outer_tail=False,
        )
        b_scale_load_operand = AsyncLoadOperand(
            context=scale_load_context,
            rsrc=fx.rocdl.get_buffer_rsrc(fx.get_iter(sb_flat)),
            lds_layout=fx.make_ordered_layout((block_n, d.scale_row_bytes), (1, 0)),
            outer_tile_size=block_n,
            outer_bound=n,
            leading_stride=scale_k,
            load_iters=d.sc_b_iters,
            is_k_major=False,
            has_outer_tail=False,
        )

    def async_load_a_to_lds(k_tile, stage):
        async_load_operand(
            a_load_operand,
            lds_base=smem_a_bytes + stage * fx.Int32(d.a_stage_bytes),
            global_outer_offset=block_m_offset,
            k_tile=k_tile,
        )

        if const_expr(d.lds_scale):
            async_load_operand(
                a_scale_load_operand,
                lds_base=smem_sca + stage * fx.Int32(d.sc_a_bytes),
                global_outer_offset=block_m_offset,
                k_tile=k_tile,
            )

    def async_load_b_to_lds(k_tile, stage):
        async_load_operand(
            b_load_operand,
            lds_base=smem_b_bytes + stage * fx.Int32(d.b_stage_bytes),
            global_outer_offset=block_n_offset,
            k_tile=k_tile,
        )
        if const_expr(d.lds_scale):
            async_load_operand(
                b_scale_load_operand,
                lds_base=smem_scb + stage * fx.Int32(d.sc_b_bytes),
                global_outer_offset=block_n_offset,
                k_tile=k_tile,
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
        if const_expr(not is_mxfp4):
            a_frag = fx.Tensor(
                fx.make_view(fx.get_iter(a_frag), fx.coalesce(a_frag.layout))
            )
            b_frag = fx.Tensor(
                fx.make_view(fx.get_iter(b_frag), fx.coalesce(b_frag.layout))
            )
        fx.gemm(
            mma_atom,
            d_frag,
            a_frag,
            b_frag,
            d_frag,
            scale_a=scale_a,
            scale_b=scale_b,
        )

    # Packed scale path. One 4-byte load holds a whole 128-element K span of
    # E8M0 scales for one row, and 64 lanes cover four MMA repeats at once.
    scale32_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Uint32)
    scale_k32 = scale_k // 4

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

    def packed_scale_issue(buf, base, row_base, repeat_stride, n_repeat, col32):
        """Issue dword scale loads and return their registers."""
        regs = []
        for q in range_constexpr(0, n_repeat, 4):
            row = row_base + fx.Int32(repeat_stride) * (fx.Int32(q) + lane_grp)
            offset = (base + row) * fx.Int32(scale_k32) + col32
            reg = fx.make_rmem_tensor(1, fx.Uint32)
            regs.append(reg)
            fx.copy(scale32_atom, fx.slice(buf, (None, offset)), reg)
        return regs

    def packed_unit_issue(buf, base, row_base, repeat_stride, n_repeat, col_base):
        """Issue dword loads over groups of four repeat/K-half units."""
        n_units = n_repeat * d.k_halves
        regs = []
        for q in range_constexpr(0, n_units, 4):
            unit = fx.Int32(q) + lane_grp
            if const_expr(q + 4 > n_units):

                unit = unit % fx.Int32(n_units)
            row = row_base + fx.Int32(repeat_stride) * (
                unit // fx.Int32(d.k_halves)
            )
            offset = (
                (base + row) * fx.Int32(scale_k32)
                + col_base
                + unit % fx.Int32(d.k_halves)
            )
            reg = fx.make_rmem_tensor(1, fx.Uint32)
            fx.copy(scale32_atom, fx.slice(buf, (None, offset)), reg)
            regs.append(reg)
        return regs

    def expand_packed_scale(packed):
        words = []
        t0, t1 = _permlane_swap(32, packed, packed)
        u0, u1 = _permlane_swap(16, t0, t0)
        w0, w1 = _permlane_swap(16, t1, t1)
        for lane_word in (u0, u1, w0, w1):
            words.append(fx.Int32(lane_word) >> (lane_grp * fx.Int32(8)))
        return words

    def packed_scale_finish(regs):
        """Exchange scale dwords among four 16-lane groups.

        Each group selects its K/32 byte from every redistributed repeat
        dword.
        """
        words = []
        for reg in regs:
            words.extend(expand_packed_scale(fx.get_scalar(reg[0]).to(fx.Int32)))
        return words

    def stage_dwords(base_bytes, stage, stage_bytes):

        ptr = base_bytes + stage * fx.Int32(stage_bytes)

        return fx.recast_iter(
            fx.PointerType.get(fx.Int32.ir_type, ptr.memspace, 16), ptr
        )

    def read_frag(base_i32, row, kh):
        """One ds_read_b128 -> i32[4]: this lane's 32 E2M1 codes for K step
        kh. Lane group g owns elements [32g, 32g+32), i.e. the single
        16-byte granule at index kh * granules_per_kh + g, XOR-swizzled
        against the row exactly as the direct-to-LDS write was."""
        granule = (fx.Int32(kh * granules_per_kh) + lane_grp) ^ (
            row & fx.Int32(granules_per_row - 1)
        )
        off = row * fx.Int32(row_dwords) + granule * fx.Int32(GFX950_DMA_BYTES // 4)
        frag = fx.make_rmem_tensor(4, fx.Int32)
        fx.copy(
            lds_copy,
            fx.make_view(fx.add_offset(base_i32, off), fx.make_layout(4, 1)),
            frag,
        )
        return frag

    def issue_frag_transposed(base_bytes, layout, row_band, kh):
        parts = []
        for part in range_constexpr(2):
            byte_block = fx.Int32(
                kh * (MXFP_MFMA_K // 2)
                + part * (MXFP_MFMA_K // 16)
            ) + lane_grp * fx.Int32(MXFP_MFMA_K // 8)
            src_kbyte = byte_block + lane_row // fx.Int32(2)
            src_outer = row_band + (lane_row % fx.Int32(2)) * fx.Int32(8)
            off = fx.get_scalar(fx.crd2idx((src_outer, src_kbyte), layout))
            addr = fx.Int32(fx.ptrtoint(base_bytes)) + fx.Int32(off)
            parts.append(_ds_read_tr8_b64(addr))
        return parts

    def finish_frag_transposed(parts):
        packed = Vec(parts[0]).shuffle(Vec(parts[1]), [0, 1, 2, 3]).ir_value()
        frag = fx.make_rmem_tensor(4, fx.Int32)
        frag.store(packed)
        return frag

    def load_fragments(stage):
        if const_expr(not is_mxfp4):
            sA_stage = fx.make_view(
                smem_a + stage * fx.Int32(block_m * block_k),
                a_lds_layout_bytes,
            )
            sB_stage = fx.make_view(
                smem_b + stage * fx.Int32(block_n * block_k),
                b_lds_layout_bytes,
            )
            thr_sA = thr_copy_A.partition_S(sA_stage)
            thr_sB = thr_copy_B.partition_S(sB_stage)
            for kh in range_constexpr(d.k_halves):
                fx.copy(
                    b_s2r_atom,
                    thr_sB[None, None, kh],
                    frag_B_retile[None, None, kh],
                )
                fx.copy(
                    a_s2r_atom,
                    thr_sA[None, None, kh],
                    frag_A_retile[None, None, kh],
                )
            return frag_A, frag_B

        base_a = stage_dwords(smem_a_bytes, stage, d.a_stage_bytes)
        base_b = stage_dwords(smem_b_bytes, stage, d.b_stage_bytes)
        base_a_bytes = smem_a_bytes + stage * fx.Int32(d.a_stage_bytes)
        base_b_bytes = smem_b_bytes + stage * fx.Int32(d.b_stage_bytes)
        av = [None] * (d.k_halves * d.mma_m_repeat)
        bv = [None] * (d.k_halves * d.mma_n_repeat)

        def _rd_b(kh, ni):
            if const_expr(b_is_transposed):
                bv[kh * d.mma_n_repeat + ni] = read_frag(
                    base_b, b_row_base + fx.Int32(ni * n_repeat_stride), kh
                )
            else:
                bv[kh * d.mma_n_repeat + ni] = issue_frag_transposed(
                    base_b_bytes,
                    b_lds_layout_bytes,
                    b_row_base - lane_row + fx.Int32(ni * n_repeat_stride),
                    kh,
                )

        def _rd_a(kh, mi):
            if const_expr(a_is_transposed):
                av[kh * d.mma_m_repeat + mi] = issue_frag_transposed(
                    base_a_bytes,
                    a_lds_layout_bytes,
                    a_row_base - lane_row + fx.Int32(mi * m_repeat_stride),
                    kh,
                )
            else:
                av[kh * d.mma_m_repeat + mi] = read_frag(
                    base_a, a_row_base + fx.Int32(mi * m_repeat_stride), kh
                )

        for kh in range_constexpr(d.k_halves):
            for ni in range_constexpr(d.mma_n_repeat):
                _rd_b(kh, ni)
            for mi in range_constexpr(d.mma_m_repeat):
                _rd_a(kh, mi)
        if const_expr(a_is_transposed or not b_is_transposed):
            rocdl.s_waitcnt(lgkmcnt=0)
            if const_expr(a_is_transposed):
                for kh in range_constexpr(d.k_halves):
                    for mi in range_constexpr(d.mma_m_repeat):
                        idx = kh * d.mma_m_repeat + mi
                        av[idx] = finish_frag_transposed(av[idx])
            if const_expr(not b_is_transposed):
                for kh in range_constexpr(d.k_halves):
                    for ni in range_constexpr(d.mma_n_repeat):
                        idx = kh * d.mma_n_repeat + ni
                        bv[idx] = finish_frag_transposed(bv[idx])
        return av, bv

    def a_fragment(frags, mi, kh):
        if const_expr(is_mxfp4):
            return frags[kh * d.mma_m_repeat + mi]
        return frags[None, mi, kh]

    def b_fragment(frags, ni, kh):
        if const_expr(is_mxfp4):
            return frags[kh * d.mma_n_repeat + ni]
        return frags[None, ni, kh]

    if const_expr(d.lds_scale):
        # Base byte offset for this lane's first scale value.
        sc_lane_base_a = (
            a_row_base * fx.Int32(d.scale_row_bytes) + lane_grp
        )
        sc_lane_base_b = (
            b_row_base * fx.Int32(d.scale_row_bytes) + lane_grp
        )

    def lds_scale_read(base_bytes, dyn_base, repeat_stride, n_repeat):
        """Read one E8M0 byte for each repeat and MFMA K slice."""
        words = []
        for r in range_constexpr(n_repeat):
            for kh in range_constexpr(d.k_halves):
                off = dyn_base + fx.Int32(
                    r * repeat_stride * d.scale_row_bytes
                    + kh * (MXFP_MFMA_K // MXFP_SCALE_BLOCK_K)
                )
                reg = fx.make_rmem_tensor(1, fx.Uint8)
                fx.copy(
                    sc_lds_atom,
                    fx.make_view(
                        fx.add_offset(base_bytes, off), fx.make_layout(1, 1)
                    ),
                    reg,
                )
                words.append(fx.get_scalar(reg[0]).to(fx.Int32))
        return words

    def compute_stage(k_tile, stage):
        if const_expr(d.lds_scale):
            av, bv = load_fragments(stage)
            sa_words = lds_scale_read(
                smem_sca,
                sc_lane_base_a + stage * fx.Int32(d.sc_a_bytes),
                m_repeat_stride,
                d.mma_m_repeat,
            )
            sb_words = lds_scale_read(
                smem_scb,
                sc_lane_base_b + stage * fx.Int32(d.sc_b_bytes),
                n_repeat_stride,
                d.mma_n_repeat,
            )
            for kh in range_constexpr(d.k_halves):
                for ni in range_constexpr(d.mma_n_repeat):
                    for mi in range_constexpr(d.mma_m_repeat):
                        scaled_mma(
                            frag_C[(None, 0), mi, ni],
                            a_fragment(av, mi, kh),
                            b_fragment(bv, ni, kh),
                            sa_words[mi * d.k_halves + kh],
                            sb_words[ni * d.k_halves + kh],
                        )
            return

        if const_expr(packed_unit_scale):
            col_base = k_tile * fx.Int32(d.k_halves)
            a_regs = packed_unit_issue(
                sa32,
                block_m_offset,
                a_row_base,
                m_repeat_stride,
                d.mma_m_repeat,
                col_base,
            )
            b_regs = packed_unit_issue(
                sb32,
                block_n_offset,
                b_row_base,
                n_repeat_stride,
                d.mma_n_repeat,
                col_base,
            )
            av, bv = load_fragments(stage)
            sa_words = packed_scale_finish(a_regs)
            sb_words = packed_scale_finish(b_regs)
            for kh in range_constexpr(d.k_halves):
                for ni in range_constexpr(d.mma_n_repeat):
                    for mi in range_constexpr(d.mma_m_repeat):
                        scaled_mma(
                            frag_C[(None, 0), mi, ni],
                            a_fragment(av, mi, kh),
                            b_fragment(bv, ni, kh),
                            sa_words[mi * d.k_halves + kh],
                            sb_words[ni * d.k_halves + kh],
                        )
            return

        if const_expr(packed_repeat_scale):
            issued = []
            for kh in range_constexpr(d.k_halves):
                col32 = k_tile * fx.Int32(d.k_halves) + fx.Int32(kh)
                issued.append(
                    (
                        packed_scale_issue(
                            sa32,
                            block_m_offset,
                            a_row_base,
                            m_repeat_stride,
                            d.mma_m_repeat,
                            col32,
                        ),
                        packed_scale_issue(
                            sb32,
                            block_n_offset,
                            b_row_base,
                            n_repeat_stride,
                            d.mma_n_repeat,
                            col32,
                        ),
                    )
                )
            av, bv = load_fragments(stage)
            for kh in range_constexpr(d.k_halves):
                sa_words = packed_scale_finish(issued[kh][0])
                sb_words = packed_scale_finish(issued[kh][1])
                for ni in range_constexpr(d.mma_n_repeat):
                    for mi in range_constexpr(d.mma_m_repeat):
                        scaled_mma(
                            frag_C[(None, 0), mi, ni],
                            a_fragment(av, mi, kh),
                            b_fragment(bv, ni, kh),
                            sa_words[mi],
                            sb_words[ni],
                        )
            return

        av, bv = load_fragments(stage)
        for kh in range_constexpr(d.k_halves):
            scale_col = (
                k_tile * fx.Int32(d.scale_row_bytes)
                + fx.Int32(kh * (MXFP_MFMA_K // MXFP_SCALE_BLOCK_K))
                + lane_grp
            )
            sa_words = [
                load_scale_word(
                    sa_flat,
                    block_m_offset + a_row_base + fx.Int32(mi * m_repeat_stride),
                    scale_col,
                )
                for mi in range_constexpr(d.mma_m_repeat)
            ]
            sb_words = [
                load_scale_word(
                    sb_flat,
                    block_n_offset + b_row_base + fx.Int32(ni * n_repeat_stride),
                    scale_col,
                )
                for ni in range_constexpr(d.mma_n_repeat)
            ]
            for ni in range_constexpr(d.mma_n_repeat):
                for mi in range_constexpr(d.mma_m_repeat):
                    scaled_mma(
                        frag_C[(None, 0), mi, ni],
                        a_fragment(av, mi, kh),
                        b_fragment(bv, ni, kh),
                        sa_words[mi],
                        sb_words[ni],
                    )

    for stage in range_constexpr(stages - 1):
        async_load_b_to_lds(fx.Int32(stage), fx.Int32(stage))
        async_load_a_to_lds(fx.Int32(stage), fx.Int32(stage))
    rocdl.sched_barrier(0)

    k_tiles = k // block_k
    main_loop_end = k_tiles - (stages - 1)
    for kt in range(0, main_loop_end, 1):
        k_tile = fx.Int32(kt)
        current_stage = k_tile % fx.Int32(stages)
        write_stage = (current_stage + fx.Int32(stages - 1)) % fx.Int32(stages)
        __barrier((stages - 2) * d.ldg_wait_count)
        next_tile = k_tile + fx.Int32(stages - 1)
        async_load_b_to_lds(next_tile, write_stage)
        async_load_a_to_lds(next_tile, write_stage)

        compute_stage(k_tile, current_stage)

    current_stage = fx.Int32(main_loop_end % stages)
    for s in range_constexpr(stages - 1):
        k_tile = fx.Int32(main_loop_end + s)
        __barrier((stages - 2 - s) * d.ldg_wait_count)
        compute_stage(k_tile, current_stage)
        current_stage = (current_stage + fx.Int32(1)) % fx.Int32(stages)

    frag_C_out = fx.make_fragment_like(frag_C, out_elem)
    frag_C_out.store(frag_C.load().to(out_elem))
    if const_expr(param.use_cshuffle):
        fx.gpu.barrier()
        for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
            row = fx.get_scalar(thr_mma_cRow[i])
            col = fx.get_scalar(thr_mma_cCol[i])
            sC[row, col] = frag_C_out[i]
        fx.gpu.barrier()
        fx.copy(cshuffle_s2r_atom, thr_sC, frag_C_cshuffle)
        fx.copy(cshuffle_r2g_atom, frag_C_cshuffle, thr_gC)
    else:
        frag_C_retile = thr_copy_C.retile(frag_C_out)
        fx.copy(r2g_atom, frag_C_retile, thr_gC)



@flyc.jit
def gemm_mxfp_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b_nk: fx.Tensor,
    scale_a_u8: fx.Tensor,
    scale_b_u8: fx.Tensor,
    param: MXFPGemmParams,
    stream: fx.Stream = fx.Stream(None),
):
    kernel = gemm_mxfp_gfx950_kernel
    kernel._known_block_size = [param.block_threads, 1, 1]
    kernel._func.__name__ = make_mxfp_gemm_kernel_name(param)
    kernel(out, a, b_nk, scale_a_u8, scale_b_u8, param).launch(
        grid=((param.m // param.block_m) * (param.n // param.block_n), 1, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )
