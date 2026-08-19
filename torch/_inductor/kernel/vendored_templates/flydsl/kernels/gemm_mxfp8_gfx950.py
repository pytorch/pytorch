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


MXFP8_SCALE_BLOCK_K = 32
MXFP8_MFMA_M = 16
MXFP8_MFMA_N = 16
MXFP8_MFMA_K = 128
GFX950_WAVE_SIZE = 64
GFX950_DMA_BYTES = 16
GFX950_LDS_CAPACITY = 163840
GFX950_NUM_XCD = 8
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
    # Asymmetric LDS: A and B may hold a different number of staged buffers.
    # None means "same as stages", which reproduces the symmetric kernel.
    stages_a: int = None
    stages_b: int = None

    def __cache_signature__(self):
        return (
            "mxfp8_gfx950_v6_asym",
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
    stages_a: int
    stages_b: int


def mxfp8_pipeline_schedule(k_tiles, stages_a, stages_b, ldg_a_iters, ldg_b_iters):
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
    a drain, instead of max(da, db) of them -- one fully unrolled 32-MFMA
    epilogue stage instead of a ragged pile of them.

    vmcnt is an in-order counter, so the barrier at the top of iteration kt may
    leave outstanding exactly the loads issued *after* the producer of the
    later of A_kt / B_kt. That count is enumerated here rather than given in
    closed form, because with da != db the "later of the two" is not always
    the same operand.
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


def mxfp8_gemm_derived(
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
) -> MXFP8GemmDerived:
    """Validate a tile config and return its derived quantities.

    Raises ValueError for any config the kernel cannot express. Mirrors
    make_gemm_gfx950_param in the FP16 kernel.

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

    if stages_b is None:
        # Give B one staged buffer more than A when every precondition holds.
        # B's DMA lands last in program order, so deepening only B is what turns
        # the K-tile boundary from a full vmcnt(0) drain into a counted wait; A
        # gains nothing from a third buffer and could not afford one anyway.
        # Worth +1.235% at 8192x8192x8192 on the champion tile (paired n=16,
        # t=4.372, p=0.00055).
        #
        # Every failure mode falls back to the symmetric depth rather than
        # raising, because raising would drop a tile config that used to be
        # valid and a shape whose autotune winner it was would silently fall
        # back to ATen. The preconditions are the extra buffer fitting in LDS,
        # K supplying more tiles than the deeper prefetch distance, and the
        # resulting waitcnt still being expressible; the last two need k, so a
        # caller that does not know the shape (the heuristics validity check)
        # gets the symmetric depth and therefore the unchanged validity set.
        stages_b = stages
        deeper = stages_a * a_stage_bytes + (stages + 1) * b_stage_bytes
        # Every check further down that could reject the deeper pipeline has to
        # be repeated here, or the fallback does not happen and a tile config
        # that was valid before is lost. The shape-independent waitcnt bound in
        # particular is looser than the one inside mxfp8_pipeline_schedule and
        # rejects configs the enumerator accepts.
        deeper_ok = (
            k is not None
            and deeper <= GFX950_LDS_CAPACITY
            and (max(stages_a, stages + 1) - 2) * ldg_wait_count < 63
        )
        if deeper_ok:
            try:
                mxfp8_pipeline_schedule(
                    k // block_k, stages_a, stages + 1, ldg_a_iters, ldg_b_iters
                )
            except ValueError:
                pass
            else:
                stages_b = stages + 1

    smem_bytes = stages_a * a_stage_bytes + stages_b * b_stage_bytes
    if smem_bytes > GFX950_LDS_CAPACITY:
        raise ValueError(
            "staged LDS buffers exceed the device shared-memory capacity: "
            f"stages_a={stages_a}, stages_b={stages_b}, block_m={block_m}, "
            f"block_n={block_n}, block_k={block_k}, smem_bytes={smem_bytes}, "
            f"capacity={GFX950_LDS_CAPACITY}"
        )
    # The exact wait counts need k_tiles, so the >= 63 check lives in
    # mxfp8_pipeline_schedule; this is the shape-independent upper bound.
    if (max(stages_a, stages_b) - 2) * ldg_wait_count >= 63:
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
        stages_a=stages_a,
        stages_b=stages_b,
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
    stages_a = gemm_config.get("STAGES_A")
    stages_b = gemm_config.get("STAGES_B")
    stages_a = None if stages_a is None else int(stages_a)
    stages_b = None if stages_b is None else int(stages_b)
    try:
        derived = mxfp8_gemm_derived(
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
        mxfp8_pipeline_schedule(
            k // block_k,
            derived.stages_a,
            derived.stages_b,
            derived.ldg_a_iters,
            derived.ldg_b_iters,
        )
    except Exception:
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
        stages_a=stages_a,
        stages_b=stages_b,
    )


def make_mxfp8_gemm_kernel_name(param: MXFP8GemmParams) -> str:
    sa = param.stages if param.stages_a is None else param.stages_a
    sb = param.stages if param.stages_b is None else param.stages_b
    return (
        "mxfp8_scaled_mm_gfx950"
        f"_{param.out_dtype}"
        f"_bm{param.block_m}_bn{param.block_n}_bk{param.block_k}"
        f"_s{param.stages}_mw{param.m_waves}_nw{param.n_waves}"
        f"_g{param.group_m}"
        f"_sa{sa}_sb{sb}"
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
    stages_a: int = None,
    stages_b: int = None,
):
    """Build a tiled gfx950 MXFP8 scaled GEMM launcher for one tile config."""
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")
    d = mxfp8_gemm_derived(
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
    )
    stages_a = d.stages_a
    stages_b = d.stages_b
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
    ) = mxfp8_pipeline_schedule(
        k_tiles, stages_a, stages_b, d.ldg_a_iters, d.ldg_b_iters
    )

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
    # Group consecutive workgroups into GROUP_M rows of the N sweep so their B
    # tiles stay hot in L2. Only exact groupings are used, which keeps the
    # index math free of a runtime min.
    use_group_m = group_m > 0 and tiles_m % group_m == 0 and tiles_m > group_m
    # The remap only helps when the swizzle is there to compact the result.
    use_xcd_remap = use_group_m
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
        if const_expr(use_xcd_remap):
            # Workgroups are handed to the 8 XCDs round-robin by id, so an id
            # written straight into the GROUP_M swizzle spreads each XCD's
            # tiles over the whole output and defeats its private L2 slice.
            # Inverting the round-robin gives every XCD a contiguous id range;
            # the swizzle below then folds that range into a compact 2-D block.
            # Both halves are needed -- without the swizzle the contiguous
            # range is a full row band, which is worse than not remapping at
            # all, so this is tied to group_m rather than enabled on its own.
            # tiles_m and tiles_n are trace-time constants, so the division and
            # remainder fold away.
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

        @fx.struct
        class SharedStorage:
            a: fx.Array[fx.Float8E4M3FN, stages_a * block_m * block_k, 16]
            b: fx.Array[fx.Float8E4M3FN, stages_b * block_n * block_k, 16]

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
        # for one row, and 64 lanes cover four MMA repeats at once. VALU
        # permlane swaps perform the 4x4 lane-group transpose that hands each
        # row's dword to the lane group that needs it; each lane then shifts out
        # its own K-quarter byte. This turns 16 single-byte loads per K-tile into
        # 4 dword loads, and it is the number of outstanding scale loads, not
        # their bytes, that the MFMA stream waits on.
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

        def packed_scale_issue(buf, base, thr_row, n_repeat, kh, col32):
            """Issue only the dword scale loads and return the landing registers.

            Split out from the transpose so the loads can be hoisted above the
            tile's direct-to-LDS DMA. vmcnt is an in-order counter, so a scale
            load issued before the DMA block is satisfied by a counted wait
            rather than the full drain a trailing load would need.
            """
            rows = [fx.get_scalar(thr_row[0, mi, kh]) for mi in range(n_repeat)]
            repeat_stride = rows[1] - rows[0]
            regs = []
            for q in range_constexpr(0, n_repeat, 4):
                row = rows[q] + repeat_stride * scale_group
                offset = (base + row) * fx.Int32(scale_k32) + col32
                reg = fx.make_rmem_tensor(1, fx.Uint32)
                fx.copy(scale32_atom, fx.slice(buf, (None, offset)), reg)
                regs.append(reg)
            return regs

        def packed_scale_finish(regs):
            """Broadcast each loaded dword's four rows across the four lane
            groups, then extract this lane's K-quarter byte.

            One dword holds a whole K-tile of scales for one row, and a wave's
            64 lanes cover four MMA repeats, so lane group g needs row g of the
            dword. Feeding the same register to both operands of a swap turns
            the general 4x4 lane-group transpose into exactly that broadcast and
            collapses four swaps into three. The VALU swaps replace ds_bpermute,
            which shares the in-order lgkmcnt with the fragment ds_reads and so
            drained them on every transpose.
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
                    # The atom above sets opsel 0 for both operands, so the byte
                    # read is byte 0 -- which is where this shift already lands
                    # the E8M0 exponent. Masking off the upper bytes and then
                    # replicating the byte across the word are therefore both
                    # dead, and dropping them shortens the dependency chain from
                    # the scale load to the MFMA. Verified by placing the byte at
                    # slot 1 with the other three zeroed: opsel 1 reproduces the
                    # reference bit for bit, opsel 0 returns garbage.
                    words.append(lane_word >> (scale_group * fx.Int32(8)))
            return words

        def load_fragments(stage_a, stage_b):
            sA_stage = fx.make_view(
                smem_a + stage_a * fx.Int32(block_m * block_k),
                a_lds_layout,
            )
            sB_stage = fx.make_view(
                smem_b + stage_b * fx.Int32(block_n * block_k),
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

        def mma_stage(k_tile, mid=None):
            """Run one K-tile's MFMAs. `mid` is the tile's fragment reads and
            direct-to-LDS DMA, run between issuing the scale loads and consuming
            them so those loads get the whole DMA block as latency shadow.
            """
            if const_expr(packed_scale):
                issued = []
                for kh in range_constexpr(d.k_halves):
                    col32 = k_tile * fx.Int32(block_k // MXFP8_MFMA_K) + fx.Int32(kh)
                    issued.append(
                        (
                            packed_scale_issue(
                                sa32, m_base, thr_mma_aRow, d.mma_m_repeat, kh, col32
                            ),
                            packed_scale_issue(
                                sb32, n_base, thr_mma_bRow, d.mma_n_repeat, kh, col32
                            ),
                        )
                    )
                if mid is not None:
                    mid()
                for kh in range_constexpr(d.k_halves):
                    sa_words = packed_scale_finish(issued[kh][0])
                    sb_words = packed_scale_finish(issued[kh][1])
                    for ni in range_constexpr(d.mma_n_repeat):
                        for mi in range_constexpr(d.mma_m_repeat):
                            scaled_mma(
                                frag_C[(None, 0), mi, ni],
                                frag_A[None, mi, kh],
                                frag_B[None, ni, kh],
                                sa_words[mi],
                                sb_words[ni],
                            )
                return

            if mid is not None:
                mid()
            for kh in range_constexpr(d.k_halves):
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

        # Prologue: iterations i in [-prologue_tiles, 0) of the same schedule
        # the steady state runs, so the vmcnt ordering is uniform from tile 0
        # on. A is filled prefetch_a deep, B prefetch_b deep, and issue order
        # is A-before-B: with prefetch_a < prefetch_b the A load is the later
        # of the two a tile waits on, so putting it first leaves B's loads
        # outstanding across the barrier instead of forcing a full drain.
        # Reversing this order collapses steady_wait back to 0.
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
            # was measured against the reverse on 12 tile configs spanning
            # 1 to 16 waves per CU and won 11 of 12 (the twelfth was a tie),
            # by 2% to 53%, so it is fixed rather than searched.
            # Both operands are issued on every trip, unconditionally, so the
            # body stays one straight-line block with one wait constant. The
            # last prefetch_b - 1 trips would address tile k_tiles or beyond;
            # the tile index is wrapped instead. The buffer tensors are built
            # with max_size=True and have no hardware bounds clamp, so the
            # wrap is what keeps the address in range. The wrapped tile lands
            # in LDS buffer (kt + prefetch_b) % stages_b, which no remaining
            # iteration reads (see mxfp8_pipeline_schedule), and its vmcnt
            # slot is accounted for by the enumerator.
            def _mid(
                cur_a=cur_a,
                cur_b=cur_b,
                k_tile=k_tile,
                write_a=write_a,
                write_b=write_b,
            ):
                load_fragments(cur_a, cur_b)
                ta = k_tile + fx.Int32(prefetch_a)
                if const_expr(wrap_a):
                    ta = ta % fx.Int32(k_tiles)
                async_load_a(ta, write_a)
                tb = k_tile + fx.Int32(prefetch_b)
                if const_expr(wrap_b):
                    tb = tb % fx.Int32(k_tiles)
                async_load_b(tb, write_b)

            mma_stage(k_tile, _mid)

        # Drain: exactly one peeled tile, the same shape the symmetric kernel
        # peels. Everything it consumes is already in flight, so it issues no
        # loads and only has to walk vmcnt down.
        kt = main_loop_end
        k_tile = fx.Int32(kt)
        cur_a = fx.Int32(kt % stages_a)
        cur_b = fx.Int32(kt % stages_b)
        __barrier(tail_waits[0])
        mma_stage(k_tile, lambda: load_fragments(cur_a, cur_b))

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
            stages_a=stages_a,
            stages_b=stages_b,
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
