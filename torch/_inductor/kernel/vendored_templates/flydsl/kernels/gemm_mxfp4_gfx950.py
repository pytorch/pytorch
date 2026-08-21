# SPDX-License-Identifier: BSD-3-Clause

"""Tile/Layout gfx950 MXFP4 scaled GEMM.

E2M1 operands use per-32-element E8M0 block scales in CDNA4 scaled 16x16x128
MFMA instructions. A is [M, K], B is [N, K], and both are row-major over K.
Two E2M1 codes share a byte, so every LDS and global address is in bytes while
every MFMA and scale index is in elements; the two units are kept in separate
names throughout (block_k vs block_k_bytes) rather than being allowed to alias
as they do in the MXFP8 kernel, where the element is a byte.

TiledMma and TiledCopy describe wave, fragment, and output ownership;
target-specific direct-to-LDS DMA and the staged waitcnt pipeline remain
explicit.
"""

import functools
from dataclasses import dataclass

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


MXFP4_SCALE_BLOCK_K = 32
MXFP4_MFMA_M = 16
MXFP4_MFMA_N = 16
MXFP4_MFMA_K = 128
# Two E2M1 codes per storage byte. This is the only source of the byte/element
# split; everywhere a quantity is derived from block_k it has to say which one
# it means.
MXFP4_ELEMS_PER_BYTE = 2
GFX950_WAVE_SIZE = 64
GFX950_DMA_BYTES = 16
# BufferCopyLDS32b moves one dword per lane. It is the widest direct-to-LDS
# copy usable for the scale tile: the 128b atom needs its 16-byte granule
# contiguous in global memory, i.e. 512 | block_k, which 6 of the 13 champion
# configs fail. (BufferCopyLDS64b must not be used -- FlyDSL 0.3.0 has no
# verifier for it, so it constructs silently and then fails instruction
# selection, i.e. the copy never happens.)
GFX950_SCALE_DMA_BYTES = 4
GFX950_LDS_CAPACITY = 163840
GFX950_NUM_XCD = 8
GFX950_MAX_BLOCK_THREADS = 1024
# The accumulator, not the fragment, bounds register blocking: a wave holds
# 4 * m_repeat * n_repeat accumulator VGPRs regardless of the operand format,
# and 8x8 alone would be 256. Halving the operand width therefore does not
# raise this cap above the value the MXFP8 kernel settled on.
MXFP4_MAX_MMA_REPEAT = 8


@dataclass(frozen=True)
class MXFP4GemmParams:
    """Compile-time identity of one specialized MXFP4 kernel.

    block_k is in *elements*; the LDS row is block_k // 2 bytes wide.
    """

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
    # Asymmetric LDS: A and B may hold a different number of staged buffers.
    # None means "same as stages", which reproduces the symmetric kernel.
    stages_a: int = None
    stages_b: int = None
    # Autotune dimension: 1 stages the E8M0 scales through LDS, 0 keeps the
    # in-register lane-group transpose. Part of the kernel identity, so the two
    # variants get separate cache entries.
    lds_scale: int = 0

    def __cache_signature__(self):
        return (
            "mxfp4_gfx950_v1",
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
        )


@dataclass(frozen=True)
class MXFP4GemmDerived:
    """Quantities derived from a tile config, shared by the kernel and the
    heuristics filter so the two can never disagree."""

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


def mxfp4_pipeline_schedule(k_tiles, stages_a, stages_b, ldg_a_iters, ldg_b_iters):
    """Program-order DMA schedule for an (stages_a, stages_b) LDS pipeline.

    Returns (main_loop_end, steady_wait, tail_waits, wrap_a, wrap_b).

    A is prefetched da = stages_a - 1 tiles ahead, B is prefetched
    db = stages_b - 1 tiles ahead. Iteration i issues A(i + da) then B(i + db);
    the prologue is iterations i in [-max(da, db), 0).

    The main loop always issues both operands so its body is one straight-line
    block with one wait constant. That means iterations near the end would run
    off the end of K, so the *tile index* of those loads is wrapped modulo
    k_tiles: they stay in bounds, land in an LDS buffer no later iteration
    reads, and are never consumed. Only their vmcnt slot matters, and it is
    accounted for below. The consequence is that exactly ONE tile is peeled as
    a drain, instead of max(da, db) of them.

    vmcnt is an in-order counter, so the barrier at the top of iteration kt may
    leave outstanding exactly the loads issued *after* the producer of the
    later of A_kt / B_kt. That count is enumerated here rather than given in
    closed form, because with da != db the "later of the two" is not always
    the same operand.

    This function is byte- and element-agnostic: it only counts load
    instructions, so it is identical to the MXFP8 kernel's.
    """
    da = stages_a - 1
    db = stages_b - 1
    deepest = max(da, db)
    main_loop_end = k_tiles - 1
    if k_tiles <= deepest:
        raise ValueError("K must supply more tiles than the deepest prefetch")
    # A wrapped load must not land on a buffer a later iteration still reads.
    # Issuing iteration i still has tiles i .. k_tiles-1 to consume, i.e. buffer
    # offsets 0 .. k_tiles-1-i <= d-1 ahead of it, while the wrapped write sits
    # d ahead; with d = stages - 1 < stages the two never alias.
    for d, s in ((da, stages_a), (db, stages_b)):
        if d >= s:
            raise ValueError("prefetch distance must be shorter than the buffer count")

    # (kind, live_tile_or_None, issuing_iteration) in program order. A wrapped
    # load carries None: it occupies a vmcnt slot but produces nothing.
    events = []
    for i in range(-deepest, 0):  # prologue
        for kind, t in (("A", i + da), ("B", i + db)):
            if t >= 0:
                events.append((kind, t, i))
    for i in range(0, main_loop_end):  # steady state: both always issued
        for kind, t in (("A", i + da), ("B", i + db)):
            events.append((kind, t if t < k_tiles else None, i))
    # The drain iteration (kt = k_tiles - 1) issues nothing.

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
    # Only emit the runtime wrap on the operand that can actually overrun.
    wrap_a = (main_loop_end - 1) + da >= k_tiles
    wrap_b = (main_loop_end - 1) + db >= k_tiles
    return main_loop_end, steady_wait, tail_waits, wrap_a, wrap_b


def mxfp4_gemm_derived(
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
) -> MXFP4GemmDerived:
    """Validate a tile config and return its derived quantities.

    block_m, block_n, block_k and k are all in elements. Raises ValueError for
    any config the kernel cannot express.

    `k` is optional and only selects the B pipeline depth; leaving it out keeps
    the result shape-independent, which is what the heuristics validity check
    wants.
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
    if block_k % MXFP4_MFMA_K != 0:
        raise ValueError(
            f"block_k must be a multiple of the MFMA K depth: block_k={block_k}"
        )
    if block_k % MXFP4_ELEMS_PER_BYTE != 0:
        raise ValueError(f"block_k must be a whole number of bytes: block_k={block_k}")
    block_k_bytes = block_k // MXFP4_ELEMS_PER_BYTE

    block_threads = m_waves * n_waves * GFX950_WAVE_SIZE
    if block_threads > GFX950_MAX_BLOCK_THREADS:
        raise ValueError(f"block exceeds {GFX950_MAX_BLOCK_THREADS} threads")

    wave_tile_m, rem_m = divmod(block_m, m_waves)
    wave_tile_n, rem_n = divmod(block_n, n_waves)
    if rem_m or rem_n:
        raise ValueError("block_m/block_n must be divisible by m_waves/n_waves")

    mma_m_repeat, rem_m = divmod(wave_tile_m, MXFP4_MFMA_M)
    mma_n_repeat, rem_n = divmod(wave_tile_n, MXFP4_MFMA_N)
    if rem_m or rem_n or mma_m_repeat == 0 or mma_n_repeat == 0:
        raise ValueError(
            "each wave tile must be a positive multiple of the 16x16 MFMA tile"
        )
    if mma_m_repeat > MXFP4_MAX_MMA_REPEAT or mma_n_repeat > MXFP4_MAX_MMA_REPEAT:
        raise ValueError(
            "accumulator repeats exceed the register budget: "
            f"mma_m_repeat={mma_m_repeat}, mma_n_repeat={mma_n_repeat}"
        )

    granules_per_row = block_k_bytes // GFX950_DMA_BYTES
    # The LDS layout XORs the granule index with the row, so the granule count
    # has to be a power of two or the swizzle would leave the row.
    if granules_per_row == 0 or granules_per_row & (granules_per_row - 1):
        raise ValueError(
            f"block_k / {2 * GFX950_DMA_BYTES} must be a power of two for the "
            f"XOR swizzle: block_k={block_k}"
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
    # The global scale path costs the loop body at both ends. Each dword load
    # serves four MMA repeats, but row_base carries lane_row, so a wave's 64
    # lanes read 64 DIFFERENT scale rows -- 64 distinct cache lines per
    # instruction, which at 64,128,512 is 67% of the loop's L1 traffic. Then a
    # three-swap permlane transpose plus a per-lane shift turns those dwords
    # into one word per (repeat, K step): 64 instructions at 128,128,512, the
    # same count as the MFMAs themselves, and they drag 24 v_mov and 15 hazard
    # s_nop along with them.
    #
    # Staging the tile's scales in LDS collapses both. The tile is fetched once
    # cooperatively by a linear direct-to-LDS DMA, and each lane then does one
    # ds_read_u8 per (repeat, K step) -- which lands the E8M0 exponent in bits
    # 7:0, exactly the byte the scaled MFMA reads at opsel 0, so there is no
    # shift, no mask and no transpose. The address is a loop-invariant per-lane
    # base plus a compile-time constant, so it folds into the instruction's
    # offset field and costs no address arithmetic either.
    #
    # It is not free: the ds_read_u8 results are live where the transpose's
    # were rematerialisable, which costs registers. Configs where that does not
    # fit fall back below rather than spilling.
    scale_row_bytes = block_k // MXFP4_SCALE_BLOCK_K
    sc_bytes_per_pass = block_threads * GFX950_SCALE_DMA_BYTES
    sc_a_bytes = block_m * scale_row_bytes
    sc_b_bytes = block_n * scale_row_bytes
    # The scale DMA has to cover its tile exactly, for the same reason the tile
    # DMA does: a partial pass would need predication the barrier accounting
    # cannot express.
    scale_dma_exact = (
        sc_a_bytes % sc_bytes_per_pass == 0 and sc_b_bytes % sc_bytes_per_pass == 0
    )

    # The pipeline depth is chosen exactly as it would be WITHOUT the scale
    # region, and the scale region is then only taken if it fits in whatever
    # headroom is left. Letting it buy space by dropping a B stage was measured
    # and is a clear loss: on the three champion configs whose tiles already sit
    # at the 163840 cap, stages_b 3 -> 2 turned a +2..+5% scale win into
    # -4.6% / -6.9% / -12.0%. The kernel's own reason is in the stages_b comment
    # above -- B's extra buffer is what makes the K-tile boundary a counted wait
    # instead of a full vmcnt(0) drain, and that is worth more than the scales.
    if stages_b is None:
        stages_b = stages
        deeper_ok = (
            k is not None
            and stages_a * a_stage_bytes + (stages + 1) * b_stage_bytes
            <= GFX950_LDS_CAPACITY
            and (max(stages_a, stages + 1) - 2) * (ldg_a_iters + ldg_b_iters) < 63
        )
        if deeper_ok:
            try:
                mxfp4_pipeline_schedule(
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

    # lds_scale_req is an autotune dimension, not a heuristic: 0 keeps the
    # in-register transpose, 1 demands the LDS path. Requesting a path the
    # config cannot express raises, so the two variants never collapse onto the
    # same kernel and the search space carries no duplicates.
    #
    # It has to be searched rather than derived. The dominant term is LDS
    # occupancy -- a scale region that pushes workgroups-per-CU from 2 to 1 cost
    # -12% to -38% -- but that is not sufficient: the SAME tile 64,64,512 gained
    # +13.0% at 256x4096x4096 and lost 11.9% at 4096^3, because at the first
    # shape the grid already limited occupancy to 1 WG/CU so the LDS ceiling
    # never bound. Two further configs regress with no occupancy change at all,
    # and mma_m_repeat does not separate the sample either (m_repeat 8 gained
    # 11.1%, m_repeat 2 lost 11.9%). Gating on a rule that does not fit the
    # measurements would be worse than letting autotune decide.
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
    # The scale DMA lengthens the in-order vmcnt chain, so the wait budget has
    # to be rechecked against it -- and given up rather than raised, since the
    # config is perfectly valid without the scale region.
    if lds_scale and (max(stages_a, stages_b) - 2) * ldg_wait_count >= 63:
        raise ValueError(
            "the scale DMA lengthens the in-order vmcnt chain past the wait "
            "budget for this pipeline depth"
        )

    smem_bytes = tile_bytes + scale_bytes
    # The exact wait counts need k_tiles, so the >= 63 check lives in
    # mxfp4_pipeline_schedule; this is the shape-independent upper bound.
    if (max(stages_a, stages_b) - 2) * ldg_wait_count >= 63:
        raise ValueError("staged pipeline wait count exceeds supported range")

    return MXFP4GemmDerived(
        block_threads=block_threads,
        block_k_bytes=block_k_bytes,
        mma_m_repeat=mma_m_repeat,
        mma_n_repeat=mma_n_repeat,
        k_halves=block_k // MXFP4_MFMA_K,
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


def make_mxfp4_param_and_validate(m, n, k, out_dtype, gemm_config):
    """Return MXFP4GemmParams for a concrete shape, or None if unsupported.

    m, n, k are in elements: k is the logical contraction length, i.e. twice
    the packed extent the tensors report.
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
    stages_a = gemm_config.get("STAGES_A")
    stages_b = gemm_config.get("STAGES_B")
    stages_a = None if stages_a is None else int(stages_a)
    stages_b = None if stages_b is None else int(stages_b)
    lds_scale = int(gemm_config.get("LDS_SCALE", 0))
    try:
        derived = mxfp4_gemm_derived(
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
    # No boundary predication: the tile must divide the problem exactly.
    if m % block_m or n % block_n or k % block_k:
        return None
    # The prologue fills max(stages_a, stages_b) - 1 buffers before the
    # steady-state loop runs.
    if (k // block_k) <= max(derived.stages_a, derived.stages_b) - 1:
        return None
    try:
        mxfp4_pipeline_schedule(
            k // block_k,
            derived.stages_a,
            derived.stages_b,
            derived.dma_a_iters,
            derived.dma_b_iters,
        )
    except Exception:
        return None
    del derived
    return MXFP4GemmParams(
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


def make_mxfp4_gemm_kernel_name(param: MXFP4GemmParams) -> str:
    sa = param.stages if param.stages_a is None else param.stages_a
    sb = param.stages if param.stages_b is None else param.stages_b
    return (
        "mxfp4_scaled_mm_gfx950"
        f"_{param.out_dtype}"
        f"_bm{param.block_m}_bn{param.block_n}_bk{param.block_k}"
        f"_s{param.stages}_mw{param.m_waves}_nw{param.n_waves}"
        f"_g{param.group_m}"
        f"_sa{sa}_sb{sb}"
        f"_ls{param.lds_scale}"
    )


# TODO: Move this common ROCm synchronization helper to FlyDSL.
def __barrier(vmcnt=0):
    llvm.InlineAsmOp(
        None,
        [],
        f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier",
        "",
        has_side_effects=True,
    )


@functools.lru_cache(maxsize=256)
def make_mxfp4_scaled_mm_gfx950(
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
    stages_a: int = None,
    stages_b: int = None,
    lds_scale: int = 0,
):
    """Build a tiled gfx950 MXFP4 scaled GEMM launcher for one tile config.

    m, n, k and block_k are in elements. The A and B tensors handed to the
    launcher are the packed uint8 views: [m, k // 2] and [n, k // 2].
    """
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")
    d = mxfp4_gemm_derived(
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
    ) = mxfp4_pipeline_schedule(
        k_tiles, stages_a, stages_b, d.dma_a_iters, d.dma_b_iters
    )

    if out_dtype == "bfloat16":
        out_elem = fx.BFloat16
    elif out_dtype == "float16":
        out_elem = fx.Float16
    else:
        raise ValueError(f"unsupported MXFP4 output dtype: {out_dtype}")

    block_threads = d.block_threads
    granules_per_row = d.granules_per_row
    swizzle_bits = granules_per_row.bit_length() - 1
    granule_byte_bits = GFX950_DMA_BYTES.bit_length() - 1
    k_bytes = k // MXFP4_ELEMS_PER_BYTE
    scale_k = k // MXFP4_SCALE_BLOCK_K
    tiles_m = m // block_m
    tiles_n = n // block_n
    grid_size = tiles_m * tiles_n
    # Group consecutive workgroups into GROUP_M rows of the N sweep so their B
    # tiles stay hot in L2. Only exact groupings are used, which keeps the
    # index math free of a runtime min.
    use_group_m = group_m > 0 and tiles_m % group_m == 0 and tiles_m > group_m
    # The remap only helps when the swizzle is there to compact the result.
    use_xcd_remap = use_group_m
    # One dword load feeds four repeats through the 4x4 lane-group transpose.
    # When the register blocking supplies four of them, the per-K-step loads sit
    # at adjacent dwords and LLVM merges them into dwordx2/x4 -- the dividend
    # recorded in MXFP4-v1-DERIVATION.md 5.1. Nothing below changes that path.
    packed_repeat_scale = d.mma_m_repeat % 4 == 0 and d.mma_n_repeat % 4 == 0
    # Shallower blocking used to fall straight back to per-BYTE scale gathers,
    # and that is where the kernel loses: every EXHAUSTIVE champion on a cell
    # that loses to Triton has a per-wave tile under 64 in some dimension, which
    # is exactly the condition above (repeat = per-wave extent / 16). Measured
    # cost was 6-10x the L1 accesses per MFMA (P0-STRUCTURAL-FINDINGS.md).
    #
    # The four lanes groups do not have to carry four *repeats*. A unit is "one
    # row's dword at one MFMA K step", and k_halves supplies units just as well,
    # so flattening (repeat, k_half) refills the groups when repeats alone
    # cannot. Those loads do not land on adjacent dwords, so they do not merge
    # -- but ceil(units/4) beats the byte path's one instruction per repeat per
    # K step, which is the comparison the gate makes.
    a_scale_units = d.mma_m_repeat * d.k_halves
    b_scale_units = d.mma_n_repeat * d.k_halves
    packed_unit_scale = not packed_repeat_scale and (
        -(-a_scale_units // 4) + -(-b_scale_units // 4)
        < d.k_halves * (d.mma_m_repeat + d.mma_n_repeat)
    )
    packed_scale = packed_repeat_scale or packed_unit_scale

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
            # Workgroups are handed to the 8 XCDs round-robin by id, so an id
            # written straight into the GROUP_M swizzle spreads each XCD's
            # tiles over the whole output and defeats its private L2 slice.
            # Inverting the round-robin gives every XCD a contiguous id range;
            # the swizzle below then folds that range into a compact 2-D block.
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

        # Sized in elements; fx.Array converts with the 4-bit element width, so
        # the allocation is block_m * block_k_bytes bytes per staged buffer.
        if const_expr(d.lds_scale):

            @fx.struct
            class SharedStorage:
                a: fx.Array[fx.Float4E2M1FN, stages_a * block_m * block_k, 16]
                b: fx.Array[fx.Float4E2M1FN, stages_b * block_n * block_k, 16]
                # E8M0 bytes, one per 32 elements, staged on the same schedule.
                sca: fx.Array[fx.Uint8, stages_a * d.sc_a_bytes, 16]
                scb: fx.Array[fx.Uint8, stages_b * d.sc_b_bytes, 16]

        else:

            @fx.struct
            class SharedStorage:
                a: fx.Array[fx.Float4E2M1FN, stages_a * block_m * block_k, 16]
                b: fx.Array[fx.Float4E2M1FN, stages_b * block_n * block_k, 16]

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        smem_a = storage.a.ptr
        smem_b = storage.b.ptr
        # Direct-to-LDS stores address bytes, and every DMA constant below
        # (granule size, wave stride, stage stride) is a byte count, so the
        # write side gets its own byte-typed iterator over the same allocation.
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

        # A and B arrive as the packed uint8 views, so their flat extents are
        # byte counts.
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
                MXFP4_MFMA_M,
                MXFP4_MFMA_N,
                MXFP4_MFMA_K,
                fx.Float4E2M1FN,
                fx.Float4E2M1FN,
                fx.Float32,
                opsel_a=0,
                opsel_b=0,
            )
        )
        # No K permutation. The scaled MFMA covers K=128 in four lane groups of
        # 32 elements; for E2M1 those 32 elements are one contiguous 16-byte
        # block, so the fragment's K order already matches the LDS row order.
        # E4M3 splits the same 32 elements into two 16-byte blocks 64 elements
        # apart, which is the only reason the MXFP8 kernel needs one.
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout(
                (m_waves, n_waves, 1),
                (n_waves, 1, 0),
            ),
        )
        thr_mma = tiled_mma.thr_slice(tid)

        # The MFMA fragment is read with explicit byte arithmetic rather than
        # through make_fragment_A / partition_S. A layout over a 4-bit element
        # type addresses one BYTE per element (FlyDSL scales an offset by
        # max(1, width // 8), the same rule fx.Array uses for its allocation
        # size), so an element-indexed LDS view has twice the intended row
        # stride and every wave reads two source rows blended together. That is
        # invisible at compile time and shows up only as a numerical error, so
        # the LDS side is modeled in dwords the way the FlyDSL mxfp4 reference
        # kernel models it, and only the MMA atom is told the element format.
        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        # Async direct-to-LDS DMA (FlyDSL #1023) when the runtime has it.
        #
        # On gfx950 this lowers to the same buffer_load ... lds instruction as the
        # synchronous CDNA3 atom; what changes is that the backend no longer
        # inserts its own vmcnt wait before the staged LDS data is read, which is
        # the unexpected vmcnt(0) raised in review on this line.
        #
        # The explicit s_waitcnt vmcnt(N) in __barrier stays. gfx950 still counts
        # this load in vmcnt -- hasAsyncMark() is true here only via
        # hasVMemToLDSLoad(); the real s_wait_asynccnt is gfx1250+ -- so the manual
        # wait remains both valid and necessary.
        #
        # Bracketing the copies with rocdl.asyncmark(), which the atom's docstring
        # suggests, was measured and rejected: asyncmark is a side-effecting meta
        # op, so it acts as a scheduling barrier and blocks sinking the DMA into
        # the MFMA block -- the ordering v7 ablation adopted. On
        # 256,256,256,2,4,2,0 it cost 24 spills and 2 extra vmcnt(0) per iteration;
        # adding wait_asyncmark on top cost 31 and 8.
        #
        # Guarded because the released FlyDSL 0.3.1 predates #1023: without this
        # the template would raise AttributeError instead of falling back.
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

        frag_C = thr_mma.make_fragment_C(gC)
        frag_C.fill(0.0)

        # Wave and lane decomposition, matching the TiledMma's wave layout
        # (m_waves, n_waves) with n_waves as the fastest mode.
        lane = fx.Int32(tid) % fx.Int32(GFX950_WAVE_SIZE)
        wave = rocdl.readfirstlane(
            fx.Int32.ir_type, fx.Int32(tid) // fx.Int32(GFX950_WAVE_SIZE)
        )
        wave_m = fx.Int32(wave) // fx.Int32(n_waves)
        wave_n = fx.Int32(wave) % fx.Int32(n_waves)
        lane_row = lane % fx.Int32(MXFP4_MFMA_M)
        lane_grp = lane // fx.Int32(MXFP4_MFMA_M)
        # A TiledMma repeats the 16x16 atom across waves, so consecutive waves
        # own 16-row stripes that interleave rather than contiguous blocks: the
        # rows a wave owns for repeat r are r * (waves * 16) + wave * 16 + ...
        # The C fragment still comes from thr_mma, so the A/B row arithmetic
        # here has to reproduce that same ownership or only the single-wave
        # case agrees.
        m_repeat_stride = m_waves * MXFP4_MFMA_M
        n_repeat_stride = n_waves * MXFP4_MFMA_N
        a_row_base = wave_m * fx.Int32(MXFP4_MFMA_M) + lane_row
        b_row_base = wave_n * fx.Int32(MXFP4_MFMA_N) + lane_row
        # 16-byte granules spanned by one 128-element MFMA K step.
        granules_per_kh = MXFP4_MFMA_K // (2 * GFX950_DMA_BYTES)
        row_dwords = block_k_bytes // 4
        r2g_atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_elem)
        thr_copy_C = fx.make_tiled_copy_C(r2g_atom, tiled_mma).get_slice(tid)
        thr_gC = thr_copy_C.partition_S(gC)

        # The scaled-MFMA state selects one E8M0 byte for each 32-element lane
        # group. The 16x16x128 A scale is 16 rows x 4 K blocks = 64 values = one
        # per lane, which depends on the MFMA shape and the 32-element block
        # granularity only -- not on the operand format.
        scale_group = (fx.Int32(tid) % fx.Int32(GFX950_WAVE_SIZE)) // fx.Int32(
            MXFP4_MFMA_N
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
            """Linear direct-to-LDS copy of one K tile's E8M0 scales.

            No swizzle: the read side gathers single bytes, so the only thing a
            swizzle would buy is bank spreading, and the write side is linear by
            construction (the DMA lays lane i at lds_ptr + i * 4).
            """
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
            # Issued immediately after the tile copy for the same K tile, which
            # is what lets the pipeline schedule just add sc_a_iters to this
            # operand's cost: vmcnt is in-order.
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
            # Output ownership still comes from the workgroup TiledMma; the A/B
            # operands are the raw i32[4] register fragments read above, and the
            # dynamic E8M0 state is bound on each underlying 16x16 atom call.
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
        # VALU permlane swaps perform the 4x4 lane-group transpose that hands
        # each row's dword to the lane group that needs it; each lane then
        # shifts out its own K-quarter byte.
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
            """Issue only the dword scale loads and return the landing registers.

            Split out from the transpose so the loads can be hoisted above the
            tile's direct-to-LDS DMA. vmcnt is an in-order counter, so a scale
            load issued before the DMA block is satisfied by a counted wait
            rather than the full drain a trailing load would need.
            """
            # A wave's 64 lanes cover 4 repeats at once: lane group g takes the
            # row of repeat q + g, so one dword load serves four repeats and the
            # permlane transpose below hands each group the row it needs.
            regs = []
            for q in range_constexpr(0, n_repeat, 4):
                row = row_base + fx.Int32(repeat_stride) * (fx.Int32(q) + scale_group)
                offset = (base + row) * fx.Int32(scale_k32) + col32
                reg = fx.make_rmem_tensor(1, fx.Uint32)
                fx.copy(scale32_atom, fx.slice(buf, (None, offset)), reg)
                regs.append(reg)
            return regs

        def packed_unit_issue(buf, base, row_base, repeat_stride, n_repeat, col_base):
            """packed_scale_issue for blocking too shallow to give four repeats.

            Unit u = mi * k_halves + kh addresses one row's dword at one MFMA K
            step; lane group g takes unit q + g. k_halves is the fast axis so a
            full load's four groups read consecutive dwords of at most two rows.
            """
            n_units = n_repeat * d.k_halves
            regs = []
            for q in range_constexpr(0, n_units, 4):
                unit = fx.Int32(q) + scale_group
                if const_expr(q + 4 > n_units):
                    # Short tail: wrap onto a valid unit rather than address off
                    # the end of the scale tensor. The surplus groups' words are
                    # simply never consumed.
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

            Feeding the same register to both operands of a swap turns the
            general 4x4 lane-group transpose into that broadcast and collapses
            four swaps into three. The VALU swaps replace ds_bpermute, which
            shares the in-order lgkmcnt with the fragment ds_reads.
            """
            words = []
            for reg in regs:
                packed = fx.get_scalar(reg[0]).to(fx.Int32)
                t0, t1 = _permlane_swap(32, packed, packed)
                u0, u1 = _permlane_swap(16, t0, t0)
                w0, w1 = _permlane_swap(16, t1, t1)
                for lane_word in (u0, u1, w0, w1):
                    lane_word = fx.Int32(lane_word)
                    # The scaled MFMA reads exactly one byte of its 32-bit scale
                    # operand, selected by opsel, and ignores the other three.
                    # The atom sets opsel 0 for both operands, so the byte read
                    # is byte 0 -- where this shift already lands the E8M0
                    # exponent. Masking off the upper bytes and replicating the
                    # byte across the word are therefore both dead.
                    words.append(lane_word >> (scale_group * fx.Int32(8)))
            return words

        def stage_dwords(base_bytes, stage, stage_bytes):
            # Offsetting a pointer advances it by elem_size bytes per unit, so
            # the stage stride is done in bytes and then recast to dwords, which
            # is the unit every fragment offset below is expressed in.
            ptr = base_bytes + stage * fx.Int32(stage_bytes)
            # Stage strides are whole 16-byte granules, so restate the alignment
            # the byte pointer lost when it was offset; a ds_read_b128 needs it.
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

        def load_fragments(stage_a, stage_b):
            base_a = stage_dwords(smem_a_bytes, stage_a, d.a_stage_bytes)
            base_b = stage_dwords(smem_b_bytes, stage_b, d.b_stage_bytes)
            av = []
            bv = []
            for kh in range_constexpr(d.k_halves):
                for ni in range_constexpr(d.mma_n_repeat):
                    bv.append(
                        read_frag(
                            base_b,
                            b_row_base + fx.Int32(ni * n_repeat_stride),
                            kh,
                        )
                    )
                for mi in range_constexpr(d.mma_m_repeat):
                    av.append(
                        read_frag(
                            base_a,
                            a_row_base + fx.Int32(mi * m_repeat_stride),
                            kh,
                        )
                    )
            return av, bv

        if const_expr(d.lds_scale):
            # row_base already carries lane_row; scale_group is this lane's
            # K-block. Both are loop-invariant, so this whole expression is
            # hoisted and every read below differs from it only by a
            # compile-time constant that folds into ds_read_u8's offset field.
            sc_lane_base_a = (
                a_row_base * fx.Int32(d.scale_row_bytes) + scale_group
            )
            sc_lane_base_b = (
                b_row_base * fx.Int32(d.scale_row_bytes) + scale_group
            )

        def lds_scale_read(base_bytes, dyn_base, repeat_stride, n_repeat):
            """Scale words indexed [repeat * k_halves + kh], one ds_read_u8 each.

            ds_read_u8 zero-extends the E8M0 byte into bits 7:0, which is the
            byte the scaled MFMA reads at opsel 0 -- so there is no shift, no
            mask, and no lane-group transpose. LLVM does not fuse adjacent byte
            reads, so the count is exactly (m_repeat + n_repeat) * k_halves.
            """
            words = []
            for r in range_constexpr(n_repeat):
                for kh in range_constexpr(d.k_halves):
                    off = dyn_base + fx.Int32(
                        r * repeat_stride * d.scale_row_bytes
                        + kh * (MXFP4_MFMA_K // MXFP4_SCALE_BLOCK_K)
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
            """Run one K-tile's MFMAs. `mid` performs the tile's fragment reads
            and direct-to-LDS DMA and returns the fragments; it runs between
            issuing the scale loads and consuming them so those loads get the
            whole DMA block as latency shadow.
            """
            if const_expr(d.lds_scale):
                av, bv = mid()
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
                    for ni in range_constexpr(d.mma_n_repeat):
                        for mi in range_constexpr(d.mma_m_repeat):
                            scaled_mma(
                                frag_C[(None, 0), mi, ni],
                                av[kh * d.mma_m_repeat + mi],
                                bv[kh * d.mma_n_repeat + ni],
                                sa_words[mi * d.k_halves + kh],
                                sb_words[ni * d.k_halves + kh],
                            )
                return

            if const_expr(packed_unit_scale):
                col_base = k_tile * fx.Int32(block_k // MXFP4_MFMA_K)
                a_regs = packed_unit_issue(
                    sa32, m_base, a_row_base, m_repeat_stride, d.mma_m_repeat,
                    col_base,
                )
                b_regs = packed_unit_issue(
                    sb32, n_base, b_row_base, n_repeat_stride, d.mma_n_repeat,
                    col_base,
                )
                av, bv = mid()
                sa_words = packed_scale_finish(a_regs)
                sb_words = packed_scale_finish(b_regs)
                for kh in range_constexpr(d.k_halves):
                    for ni in range_constexpr(d.mma_n_repeat):
                        for mi in range_constexpr(d.mma_m_repeat):
                            scaled_mma(
                                frag_C[(None, 0), mi, ni],
                                av[kh * d.mma_m_repeat + mi],
                                bv[kh * d.mma_n_repeat + ni],
                                sa_words[mi * d.k_halves + kh],
                                sb_words[ni * d.k_halves + kh],
                            )
                return

            if const_expr(packed_repeat_scale):
                issued = []
                for kh in range_constexpr(d.k_halves):
                    col32 = k_tile * fx.Int32(block_k // MXFP4_MFMA_K) + fx.Int32(kh)
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
                av, bv = mid()
                for kh in range_constexpr(d.k_halves):
                    sa_words = packed_scale_finish(issued[kh][0])
                    sb_words = packed_scale_finish(issued[kh][1])
                    for ni in range_constexpr(d.mma_n_repeat):
                        for mi in range_constexpr(d.mma_m_repeat):
                            scaled_mma(
                                frag_C[(None, 0), mi, ni],
                                av[kh * d.mma_m_repeat + mi],
                                bv[kh * d.mma_n_repeat + ni],
                                sa_words[mi],
                                sb_words[ni],
                            )
                return

            av, bv = mid()
            for kh in range_constexpr(d.k_halves):
                scale_col = (
                    k_tile * fx.Int32(block_k // MXFP4_SCALE_BLOCK_K)
                    + fx.Int32(kh * (MXFP4_MFMA_K // MXFP4_SCALE_BLOCK_K))
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
                for ni in range_constexpr(d.mma_n_repeat):
                    for mi in range_constexpr(d.mma_m_repeat):
                        scaled_mma(
                            frag_C[(None, 0), mi, ni],
                            av[kh * d.mma_m_repeat + mi],
                            bv[kh * d.mma_n_repeat + ni],
                            sa_words[mi],
                            sb_words[ni],
                        )

        # Prologue: iterations i in [-prologue_tiles, 0) of the same schedule
        # the steady state runs, so the vmcnt ordering is uniform from tile 0
        # on. Issue order is A-before-B: with prefetch_a < prefetch_b the A load
        # is the later of the two a tile waits on, so putting it first leaves
        # B's loads outstanding across the barrier instead of forcing a full
        # drain. Reversing this order collapses steady_wait back to 0.
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

            # A direct-to-LDS load is a VMEM op that writes LDS, so the compiler
            # cannot prove it does not alias a later ds_read and drains vmcnt to
            # zero in between. Reading the tile's fragments before issuing the
            # DMA removes the reason for that drain: the compiler then emits a
            # counted wait and sinks the DMA into the MFMA block. This ordering
            # was measured against the reverse on 12 MXFP8 tile configs and won
            # 11 of 12 by 2% to 53%, so it is fixed rather than searched.
            def _mid(
                cur_a=cur_a,
                cur_b=cur_b,
                k_tile=k_tile,
                write_a=write_a,
                write_b=write_b,
            ):
                frags = load_fragments(cur_a, cur_b)
                ta = k_tile + fx.Int32(prefetch_a)
                if const_expr(wrap_a):
                    ta = ta % fx.Int32(k_tiles)
                async_load_a(ta, write_a)
                tb = k_tile + fx.Int32(prefetch_b)
                if const_expr(wrap_b):
                    tb = tb % fx.Int32(k_tiles)
                async_load_b(tb, write_b)
                return frags

            mma_stage(k_tile, _mid, cur_a, cur_b)

        # Drain: exactly one peeled tile. Everything it consumes is already in
        # flight, so it issues no loads and only has to walk vmcnt down.
        kt = main_loop_end
        k_tile = fx.Int32(kt)
        cur_a = fx.Int32(kt % stages_a)
        cur_b = fx.Int32(kt % stages_b)
        __barrier(tail_waits[0])
        mma_stage(k_tile, lambda: load_fragments(cur_a, cur_b), cur_a, cur_b)

        frag_C_out = fx.make_fragment_like(frag_C, out_elem)
        frag_C_out.store(frag_C.load().to(out_elem))
        frag_C_retile = thr_copy_C.retile(frag_C_out)
        fx.copy(r2g_atom, frag_C_retile, thr_gC)

    kernel._func.__name__ = make_mxfp4_gemm_kernel_name(
        MXFP4GemmParams(
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
