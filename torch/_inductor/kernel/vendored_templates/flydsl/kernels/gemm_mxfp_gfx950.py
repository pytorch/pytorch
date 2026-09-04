# SPDX-License-Identifier: BSD-3-Clause

"""Tile/Layout gfx950 MXFP4 and MXFP8 scaled GEMM.

Both formats use per-32-element E8M0 block scales and CDNA4 16x16x128 scaled
MFMA instructions. The common kernel owns tiling, direct-to-LDS DMA, scale
loading, the staged waitcnt pipeline, and the output epilogue. A compile-time
format policy selects the operand type and fragment layout. MXFP8 stores one
E4M3 value per byte; MXFP4 stores two E2M1 values per byte, so logical K and
storage K remain distinct throughout the address calculations.
"""

import functools
from dataclasses import dataclass
from typing import Literal

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr, rocdl

from torch.utils._ordered_set import OrderedSet


def _permlane_swap(width, old, src):
    """v_permlane{16,32}_swap_b32 -> (new_old, new_src) as i32 IR values.

    Both operands are read-modify-write: the instruction exchanges row groups
    between them and returns both halves. width=32 swaps rows 2,3 of `old` with
    rows 0,1 of `src`; width=16 swaps the odd rows of `old` with the even rows
    of `src`.
    """
    i32 = ir.IntegerType.get_signless(32)
    sty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    fn = _rocdl_ops.permlane16_swap if width == 16 else _rocdl_ops.permlane32_swap
    res = fn(sty, fx.as_ir_value(old), fx.as_ir_value(src), False, False)
    return llvm.extractvalue(i32, res, [0]), llvm.extractvalue(i32, res, [1])


# Shared MXFP format and gfx950 hardware constants.
MXFPFormat = Literal["mxfp4", "mxfp8"]
MXFP_SCALE_BLOCK_K = 32
MXFP_MFMA_M = 16
MXFP_MFMA_N = 16
MXFP_MFMA_K = 128
GFX950_WAVE_SIZE = 64
GFX950_DMA_BYTES = 16
GFX950_SCALE_DMA_BYTES = 4
GFX950_LDS_CAPACITY = 163840
GFX950_SIMDS_PER_CU = 4
GFX950_VGPRS_PER_SIMD_LANE = 512
DEEPER_PIPELINE_VGPRS = 48
GFX950_NUM_XCD = 8
GFX950_MAX_BLOCK_THREADS = 1024
MXFP_MAX_MMA_REPEAT = 8


def _elements_per_byte(mxfp_format: MXFPFormat) -> int:
    if mxfp_format == "mxfp8":
        return 1
    if mxfp_format == "mxfp4":
        return 2
    raise ValueError(f"unsupported MXFP operand format: {mxfp_format}")


@dataclass(frozen=True)
class MXFPGemmParams:
    """Compile-time identity of one specialized MXFP kernel."""

    mxfp_format: MXFPFormat
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
    stages_a: int = None
    stages_b: int = None
    lds_scale: int = 0

    def __cache_signature__(self):
        return (
            "mxfp_gfx950_v1",
            self.mxfp_format,
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
            self.stages_a,
            self.stages_b,
            self.lds_scale,
        )


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
    # Tile DMA plus scale DMA, in program order. The pipeline schedule counts
    # load instructions, and the scale copies are issued immediately after the
    # tile copies for the same K tile, so they simply add to that operand's cost.
    dma_a_iters: int
    dma_b_iters: int
    ldg_wait_count: int
    lds_scale: bool
    sc_a_iters: int
    sc_b_iters: int
    sc_a_bytes: int
    sc_b_bytes: int
    scale_row_bytes: int
    a_stage_bytes: int
    b_stage_bytes: int
    smem_bytes: int
    stages_a: int
    stages_b: int


def mxfp_pipeline_schedule(k_tiles, stages_a, stages_b, ldg_a_iters, ldg_b_iters):
    """Program-order DMA schedule for an (stages_a, stages_b) LDS pipeline.

    Returns (main_loop_end, steady_wait, tail_waits, wrap_a, wrap_b).
    """
    da = stages_a - 1
    db = stages_b - 1
    deepest = max(da, db)
    main_loop_end = k_tiles - 1
    if k_tiles <= deepest:
        raise ValueError("K must supply more tiles than the deepest prefetch")
    for d, s in ((da, stages_a), (db, stages_b)):
        if d >= s:
            raise ValueError("prefetch distance must be shorter than the buffer count")

    events = []
    for i in range(-deepest, 0):  # prologue
        for kind, t in (("A", i + da), ("B", i + db)):
            if t >= 0:
                events.append((kind, t, i))
    for i in range(0, main_loop_end):  # steady state: both always issued
        for kind, t in (("A", i + da), ("B", i + db)):
            events.append((kind, t if t < k_tiles else None, i))

    cost = {"A": ldg_a_iters, "B": ldg_b_iters}
    pos = {}
    for idx, (kind, t, _i) in enumerate(events):
        if t is not None:
            pos[(kind, t)] = idx

    def wait_at(kt):
        p = max(pos[("A", kt)], pos[("B", kt)])
        return sum(cost[kind] for kind, _t, i in events[p + 1 :] if i < kt)

    waits = [wait_at(kt) for kt in range(k_tiles)]
    steady = OrderedSet(waits[:main_loop_end])
    if len(steady) != 1:
        raise ValueError(f"steady-state wait is not uniform: {sorted(steady)}")
    steady_wait = steady.pop()
    tail_waits = waits[main_loop_end:]
    if max(waits) >= 63:
        raise ValueError("staged pipeline wait count exceeds supported range")
    wrap_a = (main_loop_end - 1) + da >= k_tiles
    wrap_b = (main_loop_end - 1) + db >= k_tiles
    return main_loop_end, steady_wait, tail_waits, wrap_a, wrap_b


def mxfp_gemm_derived(
    mxfp_format: MXFPFormat,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int = 0,
    stages_a: int = None,
    stages_b: int = None,
    k: int = None,
    lds_scale_req: int = 0,
) -> MXFPGemmDerived:
    """Validate a tile config and return its derived quantities.

    block_m, block_n, block_k and k are all in elements. Raises ValueError for
    any config the kernel cannot express.
    """
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise ValueError("block_m, block_n, and block_k must be positive")
    if stages < 2:
        raise ValueError("stages must be at least 2 for the staged LDS pipeline")
    if stages_a is None:
        stages_a = stages
    if stages_a < 2 or (stages_b is not None and stages_b < 2):
        raise ValueError("stages_a and stages_b must each be at least 2")
    if m_waves <= 0 or n_waves <= 0:
        raise ValueError("m_waves and n_waves must be positive")
    if group_m < 0:
        raise ValueError("group_m must be non-negative")
    elements_per_byte = _elements_per_byte(mxfp_format)
    if block_k % MXFP_MFMA_K != 0:
        raise ValueError(
            f"block_k must be a multiple of the MFMA K depth: block_k={block_k}"
        )
    if block_k % elements_per_byte != 0:
        raise ValueError(f"block_k must be a whole number of bytes: block_k={block_k}")
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

    granules_per_row = block_k_bytes // GFX950_DMA_BYTES
    # The LDS layout XORs the granule index with the row, so the granule count
    # has to be a power of two or the swizzle would leave the row.
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
    # The optional LDS path cooperatively stages each scale tile and replaces
    # the global dword gather plus lane transpose with per-lane ds_read_u8
    # loads.
    scale_row_bytes = block_k // MXFP_SCALE_BLOCK_K
    sc_bytes_per_pass = block_threads * GFX950_SCALE_DMA_BYTES
    sc_a_bytes = block_m * scale_row_bytes
    sc_b_bytes = block_n * scale_row_bytes
    scale_dma_exact = (
        sc_a_bytes % sc_bytes_per_pass == 0 and sc_b_bytes % sc_bytes_per_pass == 0
    )
    if stages_b is None:
        stages_b = stages
        # Add the extra B stage only when its LDS footprint and estimated VGPR
        # demand fit.
        deeper_smem = stages_a * a_stage_bytes + (stages + 1) * b_stage_bytes
        wgs_per_cu = max(1, GFX950_LDS_CAPACITY // max(1, deeper_smem))
        waves_per_simd = -(
            -(m_waves * n_waves * wgs_per_cu) // GFX950_SIMDS_PER_CU
        )
        vgpr_budget = GFX950_VGPRS_PER_SIMD_LANE // max(1, waves_per_simd)
        frag_dwords = (
            (MXFP_MFMA_M * MXFP_MFMA_K // GFX950_WAVE_SIZE) // elements_per_byte // 4
        )
        vgpr_demand = 4 * mma_m_repeat * mma_n_repeat + (
            (block_k // MXFP_MFMA_K) * (mma_m_repeat + mma_n_repeat) * frag_dwords
        )
        deeper_ok = (
            k is not None
            and deeper_smem <= GFX950_LDS_CAPACITY
            and (max(stages_a, stages + 1) - 2) * (ldg_a_iters + ldg_b_iters) < 63
            and (mxfp_format != "mxfp8"
                 or vgpr_demand + DEEPER_PIPELINE_VGPRS <= vgpr_budget)
        )
        if deeper_ok:
            try:
                mxfp_pipeline_schedule(
                    k // block_k, stages_a, stages + 1, ldg_a_iters, ldg_b_iters
                )
            except ValueError:
                pass
            else:
                stages_b = stages + 1

    tile_bytes = stages_a * a_stage_bytes + stages_b * b_stage_bytes
    if tile_bytes > GFX950_LDS_CAPACITY:
        raise ValueError(
            "staged LDS buffers exceed the device shared-memory capacity: "
            f"stages_a={stages_a}, stages_b={stages_b}, block_m={block_m}, "
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
    scale_bytes = stages_a * sc_a_bytes + stages_b * sc_b_bytes if lds_scale else 0
    if lds_scale and tile_bytes + scale_bytes > GFX950_LDS_CAPACITY:
        raise ValueError(
            "LDS-staged scales do not fit beside the staged tiles: "
            f"tile_bytes={tile_bytes}, scale_bytes={scale_bytes}, "
            f"capacity={GFX950_LDS_CAPACITY}"
        )

    dma_a_iters = ldg_a_iters + sc_a_iters
    dma_b_iters = ldg_b_iters + sc_b_iters
    ldg_wait_count = dma_a_iters + dma_b_iters
    # Scale DMA contributes to the vmcnt wait budget.
    if lds_scale and (max(stages_a, stages_b) - 2) * ldg_wait_count >= 63:
        raise ValueError(
            "the scale DMA lengthens the in-order vmcnt chain past the wait "
            "budget for this pipeline depth"
        )

    smem_bytes = tile_bytes + scale_bytes
    if (max(stages_a, stages_b) - 2) * ldg_wait_count >= 63:
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
        dma_a_iters=dma_a_iters,
        dma_b_iters=dma_b_iters,
        ldg_wait_count=ldg_wait_count,
        lds_scale=lds_scale,
        sc_a_iters=sc_a_iters,
        sc_b_iters=sc_b_iters,
        sc_a_bytes=sc_a_bytes,
        sc_b_bytes=sc_b_bytes,
        scale_row_bytes=scale_row_bytes,
        a_stage_bytes=a_stage_bytes,
        b_stage_bytes=b_stage_bytes,
        smem_bytes=smem_bytes,
        stages_a=stages_a,
        stages_b=stages_b,
    )


def make_mxfp_param_and_validate(
    mxfp_format: MXFPFormat, m, n, k, out_dtype, gemm_config
):
    """Return one concrete MXFP specialization, or None if unsupported."""
    if out_dtype not in ("bfloat16", "float16"):
        return None
    block_m = int(gemm_config["TILE_M"])
    block_n = int(gemm_config["TILE_N"])
    block_k = int(gemm_config["TILE_K"])
    stages = int(gemm_config["STAGES"])
    m_waves = int(gemm_config["BLOCK_M_WARPS"])
    n_waves = int(gemm_config["BLOCK_N_WARPS"])
    group_m = int(gemm_config["GROUP_M"])
    stages_a = gemm_config.get("STAGES_A")
    stages_b = gemm_config.get("STAGES_B")
    stages_a = None if stages_a is None else int(stages_a)
    stages_b = None if stages_b is None else int(stages_b)
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
            stages_a,
            stages_b,
            k=k,
            lds_scale_req=lds_scale,
        )
    except Exception:
        return None

    if m % block_m or n % block_n or k % block_k:
        return None
    if (k // block_k) <= max(derived.stages_a, derived.stages_b) - 1:
        return None
    try:
        mxfp_pipeline_schedule(
            k // block_k,
            derived.stages_a,
            derived.stages_b,
            derived.dma_a_iters,
            derived.dma_b_iters,
        )
    except Exception:
        return None
    del derived
    return MXFPGemmParams(
        mxfp_format=mxfp_format,
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
        stages_a=stages_a,
        stages_b=stages_b,
        lds_scale=lds_scale,
    )


def make_mxfp_gemm_kernel_name(param: MXFPGemmParams) -> str:
    sa = param.stages if param.stages_a is None else param.stages_a
    sb = param.stages if param.stages_b is None else param.stages_b
    return (
        f"{param.mxfp_format}_scaled_mm_gfx950"
        f"_{param.out_dtype}"
        f"_bm{param.block_m}_bn{param.block_n}_bk{param.block_k}"
        f"_s{param.stages}_mw{param.m_waves}_nw{param.n_waves}"
        f"_g{param.group_m}"
        f"_sa{sa}_sb{sb}"
        f"_ls{param.lds_scale}"
    )


def __barrier(vmcnt=0):
    llvm.InlineAsmOp(
        None,
        [],
        f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier",
        "",
        has_side_effects=True,
    )


@functools.lru_cache(maxsize=256)
def make_mxfp_scaled_mm_gfx950(
    *,
    mxfp_format: MXFPFormat,
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
    stages_a: int = None,
    stages_b: int = None,
    lds_scale: int = 0,
):
    """Build one tiled gfx950 MXFP scaled GEMM specialization."""
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")
    elements_per_byte = _elements_per_byte(mxfp_format)
    is_mxfp4 = mxfp_format == "mxfp4"
    operand_elem = fx.Float4E2M1FN if is_mxfp4 else fx.Float8E4M3FN
    d = mxfp_gemm_derived(
        mxfp_format,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        group_m,
        stages_a,
        stages_b,
        k=k,
        lds_scale_req=lds_scale,
    )
    stages_a = d.stages_a
    stages_b = d.stages_b
    block_k_bytes = d.block_k_bytes
    prefetch_a = stages_a - 1
    prefetch_b = stages_b - 1
    prologue_tiles = max(prefetch_a, prefetch_b)
    if m % block_m or n % block_n or k % block_k:
        raise ValueError(
            f"shape must be divisible by the tile: {m}x{n}x{k} vs "
            f"{block_m}x{block_n}x{block_k}"
        )
    k_tiles = k // block_k
    if k_tiles <= prologue_tiles:
        raise ValueError("K must supply more tiles than the deepest prefetch")
    (
        main_loop_end,
        steady_wait,
        tail_waits,
        wrap_a,
        wrap_b,
    ) = mxfp_pipeline_schedule(
        k_tiles, stages_a, stages_b, d.dma_a_iters, d.dma_b_iters
    )

    if out_dtype == "bfloat16":
        out_elem = fx.BFloat16
    elif out_dtype == "float16":
        out_elem = fx.Float16
    else:
        raise ValueError(f"unsupported MXFP output dtype: {out_dtype}")

    block_threads = d.block_threads
    granules_per_row = d.granules_per_row
    swizzle_bits = granules_per_row.bit_length() - 1
    granule_byte_bits = GFX950_DMA_BYTES.bit_length() - 1
    k_bytes = k // elements_per_byte
    scale_k = k // MXFP_SCALE_BLOCK_K
    tiles_m = m // block_m
    tiles_n = n // block_n
    grid_size = tiles_m * tiles_n
    # Traverse exact groups of GROUP_M M tiles before advancing along N.
    use_group_m = group_m > 0 and tiles_m % group_m == 0 and tiles_m > group_m
    # Apply PID remapping together with GROUP_M traversal.
    use_xcd_remap = use_group_m
    # Use one dword scale load for every four repeats when both repeat counts
    # are multiples of four.
    packed_repeat_scale = d.mma_m_repeat % 4 == 0 and d.mma_n_repeat % 4 == 0

    a_scale_units = d.mma_m_repeat * d.k_halves
    b_scale_units = d.mma_n_repeat * d.k_halves
    packed_unit_scale = not packed_repeat_scale and (
        -(-a_scale_units // 4) + -(-b_scale_units // 4)
        < d.k_halves * (d.mma_m_repeat + d.mma_n_repeat)
    )
    packed_scale = packed_repeat_scale or packed_unit_scale

    # The specialized MXFP4 path keeps accumulators in AGPRs, uses a
    # boustrophedon MFMA order, and interleaves deferred fragment and DMA loads
    # with MFMA clusters.
    _prod_gate = (
        is_mxfp4
        and block_m == 256
        and block_n == 256
        and block_k == 256
        and m_waves == 2
        and n_waves == 2
        and stages == 2
        and lds_scale == 0
        and group_m in (0, 4)
        and packed_repeat_scale
    )

    ENABLE_GENERALIZED_PATH = True
    GENERALIZED_SNAKE = True
    GENERALIZED_AGPR = True
    GENERALIZED_RIFFLE_MODE = 'flat'
    GFX950_AGPRS_PER_LANE = 256
    _generalized_gate = (
        is_mxfp4
        and not packed_repeat_scale
        and (d.lds_scale or packed_unit_scale)
        and block_m * block_n >= 4096
        and stages >= 2
        and group_m in (0, 4)
        and m_waves * n_waves <= 4
        and 4 * d.mma_m_repeat * d.mma_n_repeat <= GFX950_AGPRS_PER_LANE
    )
    mxfp4_fast_path = _prod_gate or (ENABLE_GENERALIZED_PATH and _generalized_gate)

    _generalized_path = mxfp4_fast_path and not _prod_gate
    fp_snake = mxfp4_fast_path and ((not _generalized_path) or GENERALIZED_SNAKE)
    fp_agpr = mxfp4_fast_path and ((not _generalized_path) or GENERALIZED_AGPR)
    fp_riffle = "flat" if not _generalized_path else GENERALIZED_RIFFLE_MODE
    fp_block_drain = mxfp4_fast_path and fp_riffle == "none"

    def _pending_manifest():
        """Describe deferred operations in issue order."""
        man = []
        for kh in range(1, d.k_halves):
            for _ in range(d.mma_n_repeat):
                man.append(("frag", kh, kh))
            for _ in range(d.mma_m_repeat):
                man.append(("frag", kh, kh))
        n_a = d.ldg_a_iters + (1 if d.lds_scale else 0)
        n_b = d.ldg_b_iters + (1 if d.lds_scale else 0)
        for _ in range(n_a):
            man.append(("dma_a", None, d.k_halves))
        for _ in range(n_b):
            man.append(("dma_b", None, d.k_halves))
        return man

    def _spread_pending(ids, n_mfma):
        """Distribute IDs monotonically over MFMA slots."""
        n = len(ids)
        return [((j * n_mfma) // n, ids[j]) for j in range(n)]

    def _pending_plan(n_pending, n_mfma):
        """Map MFMA slots to deferred operations while respecting first use."""
        if n_pending == 0 or fp_riffle == "none":
            return {}
        man = _pending_manifest()
        assert len(man) == n_pending, (
            f"deferred issue manifest {len(man)} != live pending {n_pending}")
        cells = {}

        def _put(c, p, i):
            cells.setdefault((c, p), []).append(i)

        frag_ids = [i for i, e in enumerate(man) if e[0] == "frag"]
        dma_ids = [i for i, e in enumerate(man) if e[0] != "frag"]
        if fp_riffle == "flat":
            # Spread all pending operations through the first MFMA cluster and
            # issue any excess after its final MFMA.
            slots = sorted(set((t * n_mfma) // n_pending for t in range(n_pending)))
            for i in range(len(slots)):
                _put(0, slots[i], i)
            for i in range(len(slots), n_pending):
                _put(0, n_mfma - 1, i)
        elif fp_riffle == "staged":
            # Issue fragment reads in the latest legal preceding cluster to
            # shorten their live range.
            by_stage = {}
            for i in frag_ids:
                by_stage.setdefault(man[i][1], []).append(i)
            for kh in sorted(by_stage):
                for p, i in _spread_pending(by_stage[kh], n_mfma):
                    _put(kh - 1, p, i)
            for p, i in _spread_pending(dma_ids, n_mfma):
                _put(0, p, i)
        else:
            raise ValueError(f"unknown deferred issue mode {fp_riffle!r}")

        plan = {k: tuple(v) for k, v in cells.items()}
        seen = sorted(i for v in plan.values() for i in v)
        assert seen == list(range(n_pending)), (
            f"deferred issue plan covers {seen}; expected {n_pending} thunks")
        pos = {}
        for (c, p), v in plan.items():
            for i in v:
                pos[i] = (c, p)
        for i, (kind, stage, first_use) in enumerate(man):
            assert pos[i][0] < first_use, (
                f"deferred thunk {i} ({kind},{stage}) issued in cluster {pos[i][0]}, "
                f"first used in cluster {first_use}")
        for kind in ("frag", "dma_a", "dma_b"):
            ids = [i for i, e in enumerate(man) if e[0] == kind]
            seq = [pos[i] for i in ids]
            assert seq == sorted(seq), f"deferred {kind} thunks reordered"
        return plan
    # Register-carried scale prefetch is implemented for packed repeat scales
    # and enabled when accumulator pressure leaves no other wave to hide the
    # scale-load latency.
    sc_prefetch = (
        packed_repeat_scale
        and m_waves * n_waves <= 4
        and 4 * d.mma_m_repeat * d.mma_n_repeat >= 256
    )

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
        if const_expr(use_xcd_remap):
            # Undo PID interleaving in groups of GFX950_NUM_XCD before applying
            # the GROUP_M swizzle, giving each group a contiguous logical range.
            xcd_q, xcd_r = divmod(tiles_m * tiles_n, GFX950_NUM_XCD)
            xcd = pid % fx.Int32(GFX950_NUM_XCD)
            in_xcd = pid // fx.Int32(GFX950_NUM_XCD)
            if const_expr(xcd_r == 0):
                pid = xcd * fx.Int32(xcd_q) + in_xcd
            else:
                # branchless min(xcd, xcd_r) from the sign mask of xcd - xcd_r
                diff = xcd - fx.Int32(xcd_r)
                pid = (
                    xcd * fx.Int32(xcd_q)
                    + fx.Int32(xcd_r)
                    + (diff & (diff >> fx.Int32(31)))
                    + in_xcd
                )
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

        # Arrays are sized in logical elements; their element type determines
        # whether one or two values occupy each storage byte.
        if const_expr(d.lds_scale):

            @fx.struct
            class SharedStorage:
                a: fx.Array[operand_elem, stages_a * block_m * block_k, 16]
                b: fx.Array[operand_elem, stages_b * block_n * block_k, 16]
                # E8M0 bytes, one per 32 elements, staged on the same schedule.
                sca: fx.Array[fx.Uint8, stages_a * d.sc_a_bytes, 16]
                scb: fx.Array[fx.Uint8, stages_b * d.sc_b_bytes, 16]

        else:

            @fx.struct
            class SharedStorage:
                a: fx.Array[operand_elem, stages_a * block_m * block_k, 16]
                b: fx.Array[operand_elem, stages_b * block_n * block_k, 16]

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        smem_a = storage.a.ptr
        smem_b = storage.b.ptr

        smem_a_bytes = fx.recast_iter(fx.Uint8, storage.a.ptr)
        smem_b_bytes = fx.recast_iter(fx.Uint8, storage.b.ptr)
        if const_expr(d.lds_scale):
            smem_sca = fx.recast_iter(fx.Uint8, storage.sca.ptr)
            smem_scb = fx.recast_iter(fx.Uint8, storage.scb.ptr)

        def make_flat_buffer(tensor, elems):
            flat = fx.Tensor(
                fx.make_view(fx.get_iter(tensor), fx.make_layout(elems, 1))
            )
            return fx.rocdl.make_buffer_tensor(flat, max_size=True)

        # A and B arrive as uint8 views, so their flat extents are byte counts.
        a_flat = fx.logical_divide(
            make_flat_buffer(a, m * k_bytes), fx.make_layout(1, 1)
        )
        b_flat = fx.logical_divide(
            make_flat_buffer(b_nk, n * k_bytes), fx.make_layout(1, 1)
        )
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
        wave_layout = fx.make_layout(
            (m_waves, n_waves, 1),
            (n_waves, 1, 0),
        )
        if const_expr(is_mxfp4):
            # One E2M1 scale group is one contiguous 16-byte granule, so the
            # fragment K order already matches the packed LDS row.
            tiled_mma = fx.make_tiled_mma(mma_atom, wave_layout)
        else:
            # E4M3 uses the scaled-MFMA split-16@64 K ordering.
            mma_permutation = fx.make_tile(
                None,
                None,
                fx.make_layout(
                    (GFX950_DMA_BYTES, 2, MXFP_MFMA_K // (2 * GFX950_DMA_BYTES)),
                    (1, MXFP_MFMA_K // 2, GFX950_DMA_BYTES),
                ),
            )
            tiled_mma = fx.make_tiled_mma(mma_atom, wave_layout, mma_permutation)
        thr_mma = tiled_mma.thr_slice(tid)

        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        if const_expr(not is_mxfp4):
            s2r_layout_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(), operand_elem
            )
            s2r_atom = fx.make_copy_atom(fx.UniversalCopy128b(), operand_elem)
            thr_copy_A = fx.make_tiled_copy_A(s2r_layout_atom, tiled_mma).get_slice(tid)
            thr_copy_B = fx.make_tiled_copy_B(s2r_layout_atom, tiled_mma).get_slice(tid)
        # Prefer the async direct-to-LDS atom when available and retain the
        # synchronous fallback for FlyDSL installations without it. The gfx950
        # pipeline still uses the explicit vmcnt wait and workgroup barrier.
        if const_expr(hasattr(fx.rocdl.cdna4, "BufferLoadAsyncLDS128b")):
            dma_atom = fx.make_copy_atom(
                fx.rocdl.cdna4.BufferLoadAsyncLDS128b(), 128
            )
        else:
            dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        scale_atom = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), fx.Uint8)

        swizzle_bytes = fx.static(
            fx.SwizzleType.get(swizzle_bits, granule_byte_bits, swizzle_bits)
        )

        def make_lds_layout_bytes(rows):
            return fx.make_composed_layout(
                swizzle_bytes,
                fx.make_ordered_layout((rows, block_k_bytes), (1, 0)),
            )

        a_lds_layout_bytes = make_lds_layout_bytes(block_m)
        b_lds_layout_bytes = make_lds_layout_bytes(block_n)

        if const_expr(not is_mxfp4):
            sA = fx.make_view(smem_a, a_lds_layout_bytes)
            sB = fx.make_view(smem_b, b_lds_layout_bytes)
            frag_A = thr_mma.make_fragment_A(sA)
            frag_B = thr_mma.make_fragment_B(sB)
            frag_A_retile = thr_copy_A.retile(frag_A)
            frag_B_retile = thr_copy_B.retile(frag_B)

        frag_C = thr_mma.make_fragment_C(gC)
        _acc_ty = None
        if const_expr(mxfp4_fast_path):
            _acc_ty = fx.as_ir_value(frag_C[(None, 0), 0, 0].load()).type
        frag_C.fill(0.0)

        lane = fx.Int32(tid) % fx.Int32(GFX950_WAVE_SIZE)
        wave = rocdl.readfirstlane(
            fx.Int32.ir_type, fx.Int32(tid) // fx.Int32(GFX950_WAVE_SIZE)
        )
        wave_m = fx.Int32(wave) // fx.Int32(n_waves)
        wave_n = fx.Int32(wave) % fx.Int32(n_waves)
        lane_row = lane % fx.Int32(MXFP_MFMA_M)
        lane_grp = lane // fx.Int32(MXFP_MFMA_M)
        # A TiledMma repeats the 16x16 atom across waves, so consecutive waves
        # own 16-row stripes that interleave rather than contiguous blocks.
        m_repeat_stride = m_waves * MXFP_MFMA_M
        n_repeat_stride = n_waves * MXFP_MFMA_N
        a_row_base = wave_m * fx.Int32(MXFP_MFMA_M) + lane_row
        b_row_base = wave_n * fx.Int32(MXFP_MFMA_N) + lane_row
        # 16-byte granules spanned by one 128-element MFMA K step.
        granules_per_kh = MXFP_MFMA_K // (elements_per_byte * GFX950_DMA_BYTES)
        row_dwords = block_k_bytes // 4
        r2g_atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_elem)
        thr_copy_C = fx.make_tiled_copy_C(r2g_atom, tiled_mma).get_slice(tid)
        thr_gC = thr_copy_C.partition_S(gC)

        # The scaled-MFMA state selects one E8M0 byte for each 32-element lane
        # group. The 16x16x128 A scale is 16 rows x 4 K blocks = 64 values = one
        # per lane.
        scale_group = (fx.Int32(tid) % fx.Int32(GFX950_WAVE_SIZE)) // fx.Int32(
            MXFP_MFMA_N
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
            return fx.get_scalar(fx.crd2idx((row, col), layout)) % fx.Int32(
                block_k_bytes
            )

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
            # the composed LDS swizzle. Every quantity here is a byte count.
            lds_ptr = make_wave_lds_ptr(smem + stage * fx.Int32(stage_bytes))
            for i in range_constexpr(ldg_iters):
                lin = (fx.Int32(i * block_threads) + fx.Int32(tid)) * fx.Int32(
                    GFX950_DMA_BYTES
                )
                row = lin // fx.Int32(block_k_bytes)
                dst_col = lin % fx.Int32(block_k_bytes)
                src_col = swizzled_col(row, dst_col, layout)
                src_offset = (
                    (rows_base + row) * fx.Int32(k_bytes)
                    + k_tile * fx.Int32(block_k_bytes)
                    + src_col
                )
                src = fx.slice(gmem, (None, src_offset))
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                fx.copy(dma_atom, src, dst)
                if i < ldg_iters - 1:
                    lds_ptr = lds_ptr + fx.Int32(block_threads * GFX950_DMA_BYTES)

        if const_expr(d.lds_scale):
            sc_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)
            sc_lds_atom = fx.make_copy_atom(fx.UniversalCopy8b(), fx.Uint8)
            # Each wave writes its own 64-lane * 4-byte chunk; the hardware
            # supplies the per-lane 4-byte stride within it.
            sc_wave_offset = rocdl.readfirstlane(
                fx.Int64.ir_type,
                fx.Int64(
                    fx.Int32(tid)
                    // fx.Int32(GFX950_WAVE_SIZE)
                    * fx.Int32(GFX950_WAVE_SIZE * GFX950_SCALE_DMA_BYTES)
                ),
            )

        def async_load_scale(gmem, smem, stage_bytes, sc_iters, rows_base, k_tile,
                             stage):
            """Copy one K tile's E8M0 scales linearly into LDS."""
            lds_ptr = (
                smem + stage * fx.Int32(stage_bytes) + fx.Int32(sc_wave_offset)
            )
            for i in range_constexpr(sc_iters):
                lin = (
                    fx.Int32(i * block_threads) + fx.Int32(tid)
                ) * fx.Int32(GFX950_SCALE_DMA_BYTES)
                row = lin // fx.Int32(d.scale_row_bytes)
                col = lin % fx.Int32(d.scale_row_bytes)
                src_offset = (
                    (rows_base + row) * fx.Int32(scale_k)
                    + k_tile * fx.Int32(d.scale_row_bytes)
                    + col
                )
                fx.copy(
                    sc_dma_atom,
                    fx.slice(gmem, (None, src_offset)),
                    fx.make_view(lds_ptr, fx.make_layout(1, 1)),
                )
                if i < sc_iters - 1:
                    lds_ptr = lds_ptr + fx.Int32(
                        block_threads * GFX950_SCALE_DMA_BYTES
                    )

        def async_load_tile_step(gmem, smem, stage_bytes, ldg_iters, rows_base,
                                 k_tile, stage, layout, i):
            """One iteration of async_load_tile's loop, addressed absolutely so
            it can be emitted anywhere (the loop form advances a pointer)."""
            lds_ptr = make_wave_lds_ptr(smem + stage * fx.Int32(stage_bytes))
            lds_ptr = lds_ptr + fx.Int32(i * block_threads * GFX950_DMA_BYTES)
            lin = (fx.Int32(i * block_threads) + fx.Int32(tid)) * fx.Int32(
                GFX950_DMA_BYTES
            )
            row = lin // fx.Int32(block_k_bytes)
            dst_col = lin % fx.Int32(block_k_bytes)
            src_col = swizzled_col(row, dst_col, layout)
            src_offset = (
                (rows_base + row) * fx.Int32(k_bytes)
                + k_tile * fx.Int32(block_k_bytes)
                + src_col
            )
            fx.copy(
                dma_atom,
                fx.slice(gmem, (None, src_offset)),
                fx.make_view(lds_ptr, fx.make_layout(1, 1)),
            )

        def async_load_a_step(k_tile, stage, i):
            async_load_tile_step(
                a_flat, smem_a_bytes, d.a_stage_bytes, d.ldg_a_iters, m_base,
                k_tile, stage, a_lds_layout_bytes, i,
            )

        def async_load_b_step(k_tile, stage, i):
            async_load_tile_step(
                b_flat, smem_b_bytes, d.b_stage_bytes, d.ldg_b_iters, n_base,
                k_tile, stage, b_lds_layout_bytes, i,
            )

        def async_load_a(k_tile, stage):
            async_load_tile(
                a_flat,
                smem_a_bytes,
                d.a_stage_bytes,
                d.ldg_a_iters,
                m_base,
                k_tile,
                stage,
                a_lds_layout_bytes,
            )

            if const_expr(d.lds_scale):
                async_load_scale(
                    sa_flat, smem_sca, d.sc_a_bytes, d.sc_a_iters,
                    m_base, k_tile, stage,
                )

        def async_load_b(k_tile, stage):
            async_load_tile(
                b_flat,
                smem_b_bytes,
                d.b_stage_bytes,
                d.ldg_b_iters,
                n_base,
                k_tile,
                stage,
                b_lds_layout_bytes,
            )
            if const_expr(d.lds_scale):
                async_load_scale(
                    sb_flat, smem_scb, d.sc_b_bytes, d.sc_b_iters,
                    n_base, k_tile, stage,
                )


        def dma_thunks_a(k_tile, stage):
            # Preserve async_load_a's tile-then-scale order because vmcnt
            # accounting follows program order.
            th = [lambda i=i: async_load_a_step(k_tile, stage, i)
                  for i in range(d.ldg_a_iters)]
            if const_expr(d.lds_scale):
                th = th + [lambda: async_load_scale(
                    sa_flat, smem_sca, d.sc_a_bytes, d.sc_a_iters,
                    m_base, k_tile, stage)]
            return th

        def dma_thunks_b(k_tile, stage):
            th = [lambda i=i: async_load_b_step(k_tile, stage, i)
                  for i in range(d.ldg_b_iters)]
            if const_expr(d.lds_scale):
                th = th + [lambda: async_load_scale(
                    sb_flat, smem_scb, d.sc_b_bytes, d.sc_b_iters,
                    n_base, k_tile, stage)]
            return th

        def _drain_slots(thunks, n_mfma):
            """Spread deferred memory operations over the first MFMA cluster.

            Return a mapping from MFMA slot to one thunk index and the index of
            the first unslotted thunk. Distinct slots preserve issue order; the
            caller emits any remaining thunks at the end of the cluster.
            """
            if const_expr(not thunks):
                return {}, 0
            n_th = len(thunks)
            slots = sorted(set([(t * n_mfma) // n_th for t in range(n_th)]))
            assign = {slots[i]: i for i in range(len(slots))}
            return assign, len(slots)

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

        # AGPR-pinned MFMA used by the specialized path.
        def scaled_mma_agpr(d_frag, a_frag, b_frag, scale_a, scale_b):
            """One scaled MFMA with the accumulator pinned in AGPR.

            The result constraint is "=a" and the accumulator input is tied to
            it with ",0", so the value never leaves its AGPR between MFMAs and
            the loop needs no v_accvgpr_read/write pair per accumulation.
            """
            ops = [
                fx.as_ir_value(a_frag.load()),
                fx.as_ir_value(b_frag.load()),
                fx.as_ir_value(scale_a),
                fx.as_ir_value(scale_b),
                fx.as_ir_value(d_frag.load()),
            ]
            res = llvm.inline_asm(
                _acc_ty,
                ops,
                "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, "
                "$0, $3, $4 op_sel_hi:[0,0,0] cbsz:4 blgp:4",
                "=a,v,v,v,v,0",
                has_side_effects=True,
            )
            d_frag.store(res)

        def scaled_mma(d_frag, a_frag, b_frag, scale_a, scale_b):
            if const_expr(fp_agpr):
                scaled_mma_agpr(d_frag, a_frag, b_frag, scale_a, scale_b)
                return
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
                row = row_base + fx.Int32(repeat_stride) * (fx.Int32(q) + scale_group)
                offset = (base + row) * fx.Int32(scale_k32) + col32
                reg = fx.make_rmem_tensor(1, fx.Uint32)
                fx.copy(scale32_atom, fx.slice(buf, (None, offset)), reg)
                regs.append(reg)
            return regs

        def packed_unit_issue(buf, base, row_base, repeat_stride, n_repeat, col_base):
            """packed_scale_issue for blocking too shallow to give four repeats."""
            n_units = n_repeat * d.k_halves
            regs = []
            for q in range_constexpr(0, n_units, 4):
                unit = fx.Int32(q) + scale_group
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

        def packed_scale_finish(regs):
            """Broadcast each loaded dword's four rows across the four lane
            groups, then extract this lane's K-quarter byte.
            """
            words = []
            for reg in regs:
                packed = fx.get_scalar(reg[0]).to(fx.Int32)
                t0, t1 = _permlane_swap(32, packed, packed)
                u0, u1 = _permlane_swap(16, t0, t0)
                w0, w1 = _permlane_swap(16, t1, t1)
                for lane_word in (u0, u1, w0, w1):
                    lane_word = fx.Int32(lane_word)
                    words.append(lane_word >> (scale_group * fx.Int32(8)))
            return words

        # --- Register-carried scale prefetch ------------------------------
        # Issue each packed scale dword one K tile before it is consumed. Since
        # VMEM completion is ordered, the existing counted pipeline wait covers
        # the carried load without a same-iteration vmcnt(0).
        def make_repeat_regs(n_repeat):
            regs = []
            for _q in range_constexpr(0, n_repeat, 4):
                regs.append(fx.make_rmem_tensor(1, fx.Uint32))
            return regs

        def packed_scale_issue_into(regs, buf, base, row_base, repeat_stride,
                                    n_repeat, col32):
            """packed_scale_issue writing into caller-owned registers."""
            for q in range_constexpr(0, n_repeat, 4):
                row = row_base + fx.Int32(repeat_stride) * (
                    fx.Int32(q) + scale_group
                )
                offset = (base + row) * fx.Int32(scale_k32) + col32
                fx.copy(scale32_atom, fx.slice(buf, (None, offset)), regs[q // 4])

        def packed_scale_read(regs):
            return [fx.get_scalar(reg[0]).to(fx.Int32) for reg in regs]

        def packed_scale_words(vals):
            """packed_scale_finish's transpose half, on already-read dwords."""
            words = []
            for packed in vals:
                t0, t1 = _permlane_swap(32, packed, packed)
                u0, u1 = _permlane_swap(16, t0, t0)
                w0, w1 = _permlane_swap(16, t1, t1)
                for lane_word in (u0, u1, w0, w1):
                    lane_word = fx.Int32(lane_word)
                    words.append(lane_word >> (scale_group * fx.Int32(8)))
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

        def load_fragments(stage_a, stage_b, defer=False):
            if const_expr(not is_mxfp4):
                sA_stage = fx.make_view(
                    smem_a + stage_a * fx.Int32(block_m * block_k),
                    a_lds_layout_bytes,
                )
                sB_stage = fx.make_view(
                    smem_b + stage_b * fx.Int32(block_n * block_k),
                    b_lds_layout_bytes,
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
                return frag_A, frag_B, []

            base_a = stage_dwords(smem_a_bytes, stage_a, d.a_stage_bytes)
            base_b = stage_dwords(smem_b_bytes, stage_b, d.b_stage_bytes)
            av = [None] * (d.k_halves * d.mma_m_repeat)
            bv = [None] * (d.k_halves * d.mma_n_repeat)
            thunks = []

            def _rd_b(kh, ni):
                bv[kh * d.mma_n_repeat + ni] = read_frag(
                    base_b, b_row_base + fx.Int32(ni * n_repeat_stride), kh
                )

            def _rd_a(kh, mi):
                av[kh * d.mma_m_repeat + mi] = read_frag(
                    base_a, a_row_base + fx.Int32(mi * m_repeat_stride), kh
                )

            for kh in range_constexpr(d.k_halves):
                if const_expr(defer and kh > 0):
                    for ni in range_constexpr(d.mma_n_repeat):
                        thunks.append(lambda kh=kh, ni=ni: _rd_b(kh, ni))
                    for mi in range_constexpr(d.mma_m_repeat):
                        thunks.append(lambda kh=kh, mi=mi: _rd_a(kh, mi))
                    continue
                for ni in range_constexpr(d.mma_n_repeat):
                    _rd_b(kh, ni)
                for mi in range_constexpr(d.mma_m_repeat):
                    _rd_a(kh, mi)
            return av, bv, thunks

        def _mfma_order():
            """Return the MFMA emission order for this wave's repeat grid."""
            m_rep, n_rep = d.mma_m_repeat, d.mma_n_repeat
            if const_expr(not fp_snake):
                return [(mi, ni) for ni in range(n_rep) for mi in range(m_rep)]
            order, seen = [], set()
            j0s = list(range(0, n_rep - n_rep % 2, 2))
            for nth, i0 in enumerate(range(0, m_rep - m_rep % 2, 2)):
                for j0 in (list(reversed(j0s)) if nth % 2 else j0s):
                    for di in range(2):
                        for dj in range(2):
                            order.append((i0 + di, j0 + dj))
            seen = set(order)
            # Odd leftovers (a repeat count that is not even) keep the default
            # order.
            order += [(mi, ni) for ni in range(n_rep) for mi in range(m_rep)
                      if (mi, ni) not in seen]
            return order

        _MMA_ORDER = _mfma_order()

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
                a_row_base * fx.Int32(d.scale_row_bytes) + scale_group
            )
            sc_lane_base_b = (
                b_row_base * fx.Int32(d.scale_row_bytes) + scale_group
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

        def mma_stage(k_tile, mid, cur_a, cur_b):
            """Run one K tile and issue every deferred operation before use.

            Depending on the scale path, pending operations are either drained
            as a block or distributed over MFMA slots by the issue plan.
            """
            if const_expr(d.lds_scale):
                av, bv, _pending = mid()
                # Select exactly one drain mode: block issue or planned issue
                # between MFMAs.
                if const_expr(fp_block_drain):
                    for _th in _pending:
                        _th()
                _plan = _pending_plan(len(_pending), len(_MMA_ORDER))
                sa_words = lds_scale_read(
                    smem_sca,
                    sc_lane_base_a + cur_a * fx.Int32(d.sc_a_bytes),
                    m_repeat_stride,
                    d.mma_m_repeat,
                )
                sb_words = lds_scale_read(
                    smem_scb,
                    sc_lane_base_b + cur_b * fx.Int32(d.sc_b_bytes),
                    n_repeat_stride,
                    d.mma_n_repeat,
                )
                for kh in range_constexpr(d.k_halves):
                    for _t in range_constexpr(len(_MMA_ORDER)):
                        mi, ni = _MMA_ORDER[_t]
                        scaled_mma(
                            frag_C[(None, 0), mi, ni],
                            a_fragment(av, mi, kh),
                            b_fragment(bv, ni, kh),
                            sa_words[mi * d.k_halves + kh],
                            sb_words[ni * d.k_halves + kh],
                        )
                        if const_expr((kh, _t) in _plan):
                            for _q in range_constexpr(len(_plan[(kh, _t)])):
                                _pending[_plan[(kh, _t)][_q]]()
                return

            if const_expr(packed_unit_scale):
                col_base = k_tile * fx.Int32(block_k // MXFP_MFMA_K)
                a_regs = packed_unit_issue(
                    sa32, m_base, a_row_base, m_repeat_stride, d.mma_m_repeat,
                    col_base,
                )
                b_regs = packed_unit_issue(
                    sb32, n_base, b_row_base, n_repeat_stride, d.mma_n_repeat,
                    col_base,
                )
                av, bv, _pending = mid()
                # Select exactly one drain mode, as in the LDS-scale branch.
                if const_expr(fp_block_drain):
                    for _th in _pending:
                        _th()
                _plan = _pending_plan(len(_pending), len(_MMA_ORDER))
                sa_words = packed_scale_finish(a_regs)
                sb_words = packed_scale_finish(b_regs)
                for kh in range_constexpr(d.k_halves):
                    for _t in range_constexpr(len(_MMA_ORDER)):
                        mi, ni = _MMA_ORDER[_t]
                        scaled_mma(
                            frag_C[(None, 0), mi, ni],
                            a_fragment(av, mi, kh),
                            b_fragment(bv, ni, kh),
                            sa_words[mi * d.k_halves + kh],
                            sb_words[ni * d.k_halves + kh],
                        )
                        if const_expr((kh, _t) in _plan):
                            for _q in range_constexpr(len(_plan[(kh, _t)])):
                                _pending[_plan[(kh, _t)][_q]]()
                return

            if const_expr(packed_repeat_scale):
                issued = []
                for kh in range_constexpr(d.k_halves):
                    col32 = k_tile * fx.Int32(block_k // MXFP_MFMA_K) + fx.Int32(kh)
                    issued.append(
                        (
                            packed_scale_issue(
                                sa32,
                                m_base,
                                a_row_base,
                                m_repeat_stride,
                                d.mma_m_repeat,
                                col32,
                            ),
                            packed_scale_issue(
                                sb32,
                                n_base,
                                b_row_base,
                                n_repeat_stride,
                                d.mma_n_repeat,
                                col32,
                            ),
                        )
                    )
                av, bv, _pending = mid()
                # Drain deferred fragment reads and DMA during the first MFMA
                # cluster. Emit leftovers before later K halves consume them.
                _slots, _left = _drain_slots(_pending, len(_MMA_ORDER))
                for kh in range_constexpr(d.k_halves):
                    sa_words = packed_scale_finish(issued[kh][0])
                    sb_words = packed_scale_finish(issued[kh][1])
                    for _t in range_constexpr(len(_MMA_ORDER)):
                        mi, ni = _MMA_ORDER[_t]
                        scaled_mma(
                            frag_C[(None, 0), mi, ni],
                            a_fragment(av, mi, kh),
                            b_fragment(bv, ni, kh),
                            sa_words[mi],
                            sb_words[ni],
                        )
                        if const_expr(kh == 0 and _t in _slots):
                            _pending[_slots[_t]]()
                    if const_expr(kh == 0):
                        for _ti in range_constexpr(_left, len(_pending)):
                            _pending[_ti]()
                return

            av, bv, _pending = mid()
            for _th in _pending:
                _th()
            for kh in range_constexpr(d.k_halves):
                scale_col = (
                    k_tile * fx.Int32(block_k // MXFP_SCALE_BLOCK_K)
                    + fx.Int32(kh * (MXFP_MFMA_K // MXFP_SCALE_BLOCK_K))
                    + scale_group
                )
                sa_words = [
                    load_scale_word(
                        sa_flat,
                        m_base + a_row_base + fx.Int32(mi * m_repeat_stride),
                        scale_col,
                    )
                    for mi in range_constexpr(d.mma_m_repeat)
                ]
                sb_words = [
                    load_scale_word(
                        sb_flat,
                        n_base + b_row_base + fx.Int32(ni * n_repeat_stride),
                        scale_col,
                    )
                    for ni in range_constexpr(d.mma_n_repeat)
                ]
                for _t in range_constexpr(len(_MMA_ORDER)):
                    mi, ni = _MMA_ORDER[_t]
                    scaled_mma(
                        frag_C[(None, 0), mi, ni],
                        a_fragment(av, mi, kh),
                        b_fragment(bv, ni, kh),
                        sa_words[mi],
                        sb_words[ni],
                    )

        sc_car_a = []
        sc_car_b = []
        for _kh in range_constexpr(d.k_halves):
            sc_car_a.append(make_repeat_regs(d.mma_m_repeat))
            sc_car_b.append(make_repeat_regs(d.mma_n_repeat))

        def sc_issue(k_tile):
            for kh in range_constexpr(d.k_halves):
                col32 = k_tile * fx.Int32(block_k // MXFP_MFMA_K) + fx.Int32(kh)
                packed_scale_issue_into(
                    sc_car_a[kh], sa32, m_base, a_row_base, m_repeat_stride,
                    d.mma_m_repeat, col32,
                )
                packed_scale_issue_into(
                    sc_car_b[kh], sb32, n_base, b_row_base, n_repeat_stride,
                    d.mma_n_repeat, col32,
                )

        def sc_read():
            va, vb = [], []
            for kh in range_constexpr(d.k_halves):
                va.append(packed_scale_read(sc_car_a[kh]))
                vb.append(packed_scale_read(sc_car_b[kh]))
            return va, vb

        # The prefetched path mirrors the packed-repeat MFMA order and drain
        # schedule; carried register values replace only the scale loads.
        def mma_stage_pf(mid, sav, sbv):
            av, bv, _pending = mid()
            _slots, _left = _drain_slots(_pending, len(_MMA_ORDER))
            for kh in range_constexpr(d.k_halves):
                sa_words = packed_scale_words(sav[kh])
                sb_words = packed_scale_words(sbv[kh])
                for _t in range_constexpr(len(_MMA_ORDER)):
                    mi, ni = _MMA_ORDER[_t]
                    scaled_mma(
                        frag_C[(None, 0), mi, ni],
                        a_fragment(av, mi, kh),
                        b_fragment(bv, ni, kh),
                        sa_words[mi],
                        sb_words[ni],
                    )
                    if const_expr(kh == 0 and _t in _slots):
                        _pending[_slots[_t]]()
                if const_expr(kh == 0):
                    for _ti in range_constexpr(_left, len(_pending)):
                        _pending[_ti]()

        if const_expr(sc_prefetch):
            sc_issue(fx.Int32(0))

        for i in range_constexpr(-prologue_tiles, 0):
            if const_expr(0 <= i + prefetch_a):
                ta = i + prefetch_a
                async_load_a(ta, fx.Int32(ta % stages_a))
            if const_expr(0 <= i + prefetch_b):
                tb = i + prefetch_b
                async_load_b(tb, fx.Int32(tb % stages_b))
        rocdl.sched_barrier(0)

        for kt in range(0, main_loop_end, 1):
            k_tile = fx.Int32(kt)
            cur_a = k_tile % fx.Int32(stages_a)
            cur_b = k_tile % fx.Int32(stages_b)
            write_a = (k_tile + fx.Int32(prefetch_a)) % fx.Int32(stages_a)
            write_b = (k_tile + fx.Int32(prefetch_b)) % fx.Int32(stages_b)
            __barrier(steady_wait)

            # Read current fragments before reusing LDS stages for later tiles.
            def _mid(
                cur_a=cur_a,
                cur_b=cur_b,
                k_tile=k_tile,
                write_a=write_a,
                write_b=write_b,
            ):
                # Specialized path: defer later-K-half fragment reads and
                # next-tile DMA so mma_stage can interleave them with MFMAs.
                av, bv, _pend = load_fragments(cur_a, cur_b,
                                               defer=mxfp4_fast_path)
                ta = k_tile + fx.Int32(prefetch_a)
                if const_expr(wrap_a):
                    ta = ta % fx.Int32(k_tiles)
                if const_expr(mxfp4_fast_path):
                    _pend = _pend + dma_thunks_a(ta, write_a)
                else:
                    async_load_a(ta, write_a)
                tb = k_tile + fx.Int32(prefetch_b)
                if const_expr(wrap_b):
                    tb = tb % fx.Int32(k_tiles)
                if const_expr(mxfp4_fast_path):
                    _pend = _pend + dma_thunks_b(tb, write_b)
                else:
                    async_load_b(tb, write_b)
                return av, bv, _pend

            if const_expr(sc_prefetch):

                sav, sbv = sc_read()
                sc_issue(k_tile + fx.Int32(1))

                rocdl.sched_barrier(0)
                mma_stage_pf(_mid, sav, sbv)
            else:
                mma_stage(k_tile, _mid, cur_a, cur_b)

        kt = main_loop_end
        k_tile = fx.Int32(kt)
        cur_a = fx.Int32(kt % stages_a)
        cur_b = fx.Int32(kt % stages_b)
        __barrier(tail_waits[0])
        if const_expr(sc_prefetch):
            sav, sbv = sc_read()
            mma_stage_pf(lambda: load_fragments(cur_a, cur_b), sav, sbv)
        else:
            mma_stage(k_tile, lambda: load_fragments(cur_a, cur_b), cur_a, cur_b)

        if const_expr(fp_agpr):
            llvm.InlineAsmOp(None, [], "s_nop 7", "", has_side_effects=True)
            rocdl.sched_barrier(0)
        frag_C_out = fx.make_fragment_like(frag_C, out_elem)
        frag_C_out.store(frag_C.load().to(out_elem))
        frag_C_retile = thr_copy_C.retile(frag_C_out)
        fx.copy(r2g_atom, frag_C_retile, thr_gC)

    kernel._func.__name__ = make_mxfp_gemm_kernel_name(
        MXFPGemmParams(
            mxfp_format=mxfp_format,
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
            stages_a=stages_a,
            stages_b=stages_b,
            lds_scale=lds_scale,
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
