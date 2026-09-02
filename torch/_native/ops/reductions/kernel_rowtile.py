# ROW reductions: launch policy for tile.TileReduce on the row axis (contiguous last dim of an
# (M, N) input). The body is in tile.py; this module owns the measured launch shapes, the narrow-row
# gates and the plan cache. Serves 1 or 2 outputs, final or raw stage-1 partials, with a ROLLED
# chunk loop, so one compiled kernel covers every N in a vec class.

import math
from typing import NamedTuple

from cutlass import Int32

import torch

from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan
from .._cutedsl.traits import WARP
from . import tile


_compile = _L.compile_kernel
_stream = _L.stream
_CACHE = {}


# --- Row-reduce occupancy heuristic, parameterized as DATA (see row_config) ---
# threads-per-row (tpr) ladder: the first (eff_N <= limit) row wins; above the last,
# TPR_MAX. Small N -> 1 warp/row so many rows pack per block with no cross-warp reduce.
# Autotuner overrides. The N-limits are B200-tuned anchors; row_config scales them by hw so
# the same ladder generalizes across GPUs (identity on B200).
_TPR_LADDER = ((64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128))
_TPR_MAX = 256
# Every legal tpr, widest last. All powers of two: tpr sets the width of the cross-thread
# reduce tree (and, when it doubles as the block size, the warp count), both of which
# require a power of two -- see single_row_config.
_TPR_RUNGS = tuple(t for _, t in _TPR_LADDER) + (_TPR_MAX,)
# threads-per-block (nt) gate: small rows use a smaller block, wider rows the larger.
_NT_SMALL, _NT_LARGE, _NT_GATE_N = 128, 256, 16 * 1024
# Wide-row fast rung: once a ROW is >= this many BYTES it needs the full 256 threads/row
# (+256/block) -- the element ladder above is dtype-blind and under-threads here. Measured
# (graph-mode sweep): tpr256/nt256 wins from row_bytes >= 16KB for BOTH fp32 (N>=4096)
# and bf16 (N>=8192) -- a clean byte-aligned crossover, 1.1-1.4x vs the ladder's tpr64/128
# -- and loses badly below it (small N wants many rows/block). Bytes, not elements, so it
# is dtype-correct without a per-dtype table.
_WIDE_ROW_BYTES = 16 * 1024


# --- NARROW rows: tpr == 1, one thread per row ---
# `tpr` floors at a WARP whenever lanes are merged, so a row of `N // vec` chunks leaves
# `WARP - chunks` lanes idle: 25% lane utilization at N=32, where the packed shape measured 4.0x
# slower than tpr == 1 at (1048576, 32). tpr == 1 merges nothing, so it serves any trait.
#
# SAFETY ceiling on the width it will take (the whole row is one thread's unroll), derived from
# MAX_UNROLL so the two cannot drift. The measured crossover is far below it -- see `narrow_row`.
_MAX_NARROW_N = min(256, tile.MAX_UNROLL)
# MEASURED ladder of (minimum rows, per-thread chunk budget): one thread per row shrinks the grid
# ~tpr times, so it needs enough rows to fill the SMs, and the wider the row the more rows. Bounds
# are in vec-CHUNKS, not raw N, so they carry across dtypes. Speedup range over the packed shape,
# B200, nfields 1/2/3:
#   M=4096 at 6 chunks 1.14-1.87x   M=16384 at 16 1.32-1.60x   M=262144 at 4 9.25-33.7x
# Tiered rather than one bound because a budget that suits M=65536 loses at M=16384.
_CHUNK_LADDER = ((65536, 32), (16384, 16), (4096, 6))

# TMA-STAGED LOAD, for the one regime where the direct load is not at SOL. Thread t reads row t, so
# the lane stride is N*itemsize and the direct load runs at 91-93% of peak only while >= 2 lanes
# share a 128-byte line: 7001 GB/s at N=16 (64 B stride) against 4584 at N=32 (128 B). It is
# OVER-FETCH, not latency, so the fix is a contiguous access, which a (rows, N) TMA box is by
# construction. Worth 1.49-1.86x, reaching 90% of peak at (4194304, 32), but ONLY with the smem
# rotation (tile.fold_smem_rotated) -- without it, a regression. Gated to power-of-two fp32 N,
# which is what the rotation's mask and bank arithmetic assume. Staged at depth 1.
_TMA_MIN_STRIDE = 128


def narrow_row(N: int, itemsize: int, M: int) -> bool:
    """Is this geometry in the regime where one thread per row beats the packed shape?"""
    if N < 1 or N > _MAX_NARROW_N:
        return False
    chunks = N // tile.vec_size(N, itemsize)
    for min_rows, budget in _CHUNK_LADDER:
        if M >= min_rows:
            return chunks <= budget
    return False


def tma_ok(N: int, itemsize: int, M: int, device=None) -> bool:
    """Should this geometry stage its load through TMA rather than load direct?"""
    if itemsize != 4 or N <= 0 or N & (N - 1) or N * itemsize < _TMA_MIN_STRIDE:
        return False
    if not narrow_row(N, itemsize, M):
        return False
    if device is not None:
        # Through the memoized caps, not get_device_properties: reduce_row_tile evaluates
        # this on EVERY launch of the band this path serves, ahead of the plan-cache
        # lookup, and the raw query costs ~1.3us (see kernel_xcta's _DEV_SM).
        from .._cutedsl import hw_caps as _hw

        if _hw.caps(device).cc[0] < 9:
            return False  # TMA is sm_90+
    return True


# --- INNER-TREE ORDER (opt-in) ---
# Every other order here derives its add association from the LAUNCH SHAPE, so it moves with tpr, nt
# or M. This one fixes the DAG from N alone -- stride-doubling tree per load, streaming carry at most
# `depth` deep, warp-major chunks, ascending butterfly across lanes -- which is what makes it
# hash-pinnable and so usable for a determinism claim. It covers EVERY N in three shapes (see
# _ItreePlan); total coverage is required, since a shape it skipped would silently keep the
# launch-shape order.
#
# OPT-IN because it costs: 0.92-1.41x the device time of the rolled fold at a fixed 256 MiB
# footprint, winning where the rolled fold underfills or its gcd `vec` collapses to scalar loads,
# losing ~1.3x on mid-width rows. It also gives up the N-free compile key.
#
# Our OWN env var, not upstream's PYTORCH_SUM_INNER_TREE: upstream's kernels register first and keep
# their eligible calls, so the two must be switchable independently and stay comparable.
_INNER_TREE_ENV = "PYTORCH_NATIVE_INNER_TREE"
# Block size for the two ONE-THREAD-PER-ROW shapes (upstream's kMultiRowThreads and
# kAccumulateThreads, both 128), and the multirow carry's cap, which is fixed rather than
# plan-derived (kMaxDepth).
_MULTIROW_ROWS_PER_BLOCK = 128
_MULTIROW_MAX_DEPTH = 6

# Per-thread width for the order, in ELEMENTS (k*vec*eff live in registers at once). Width is bought
# by fusing adjacent CHUNKS, which keeps a full warp -- and so coalesced loads -- on each: worth
# 1.79x at (65536, 1024). 64 is within 2% of the best point at every shape measured, and the shape
# falls off a cliff at 256 (127.6us against 56.7 at N=100000).
_ITREE_THREAD_ELEMS = 64

# Vector-width multiplier for the order's `vec`. 1: a wider thread run strides the lanes by its
# width, which costs ~2x per doubling. Bit-exact at any value -- `bte` pins the batch decomposition,
# which is what fixes the bits.
_ITREE_VEC_MUL = 1

# Target BLOCK SIZE, which is how rows-per-block is picked. DAG-free (only independent rows share a
# block), so this is pure occupancy. 256, because a 64-thread block starves the SM of rows at small
# N: (524288, 128) 145.3 vs 68.2us, (262144, 256) 74.0 vs 50.8.
_ITREE_BLOCK_THREADS = 256

# SMEM-STAGED FOLD, on for NARROW per-lane runs. See tile.TileReduce._fold_itree_smem. A coalesced
# load leaves a lane owning only `vec` columns while in-register tree levels need a contiguous run;
# staging breaks the tie, so one butterfly replaces span/(WARP*vec) of them. Bit-neutral.
#
# cp.async, one batch, in TILES of `_ITREE_STAGE_E * WARP` columns. MEASURED device us, chunk fusion
# -> tiled staging, with the ratio against the unordered rolled fold, which closes the mid-band's
# 1.28x deficit: (65536,1024) 48.2 -> 40.9 (1.06x), (32768,2048) 47.6 -> 38.6 (0.97x, faster),
# (16384,4096) 48.9 -> 39.2, (8192,8192) 48.7 -> 39.6. Gated on cp.async, on a single batch, and on
# the FULL per-lane run: a shorter run, a register-staged copy and an untiled buffer each measured
# worse.

# Columns per lane PER TILE. 32 is where the untiled sweep bottomed out (40.7us at N=1024); tiling
# holds every wider batch at that same per-lane run and the same 4.6 KB of smem per row.
_ITREE_STAGE_E = 32


def inner_tree_order_enabled() -> bool:
    """Is the reproducible-DAG order requested? Read live, so tests can toggle it."""
    import os

    return os.environ.get(_INNER_TREE_ENV, "") not in ("", "0")


class _ItreePlan(NamedTuple):
    """Everything the order's DAG depends on, all compile-time. One of three SHAPES, chosen from N as
    upstream's host dispatch chooses between its three kernels -- the shape is part of the DAG, so it
    cannot be a launch-time preference:

    "multirow"  one thread per row, the whole row in one static fragment (N <= vec * 8)
    "looped"    `wpr` warps per row, in up to three compile-time batches with a tile each
    "split"     one block per (row, batch) writing a partial, then a linear per-row combine

    `batches` holds (offset, remaining, loads, warp_chunk) per batch and `tms` a warp-major tile
    each. The split shape's batch index is a RUNTIME value, so its two entries are the two distinct
    WIDTHS -- a full batch and the short last one -- and `split` says what selects between them.
    """

    shape: str
    vec: int
    wpr: int  # warps cooperating on one row; 0 == one thread per row
    rows_per_block: int
    depth: int
    batches: tuple
    tms: tuple
    # split only: (nbatch, batch_total_elements, last_remaining, chunk_full, chunk_last)
    split: tuple = ()
    # ADJACENT CHUNKS one thread group folds. NOT part of the DAG: the chunks and their trees are
    # unchanged, and the k results combine locally exactly where the cross-chunk merge would have
    # combined them. Must divide wpr.
    kchunk: int = 1
    # Fold a thread's run as one LINEAR chain instead of a tree. DOES change the DAG.
    vec_linear: bool = False
    # SMEM-STAGED fold: columns per lane, so one warp folds the batch with ONE butterfly. 0 = off.
    # Bit-neutral (32 wide contiguous lanes + one butterfly == ATen's per-chunk nesting).
    stage_e: int = 0

    @property
    def sig(self):
        return (
            self.shape,
            self.vec,
            self.wpr,
            self.rows_per_block,
            self.depth,
            self.batches,
            tuple(t.sig for t in self.tms),
            self.split,
            self.kchunk,
            self.vec_linear,
            self.stage_e,
        )


def _fuse_factor(want: int | None, wpr: int, vec: int, loads: int) -> int:
    """Adjacent chunks per thread group: as many as the register budget and wpr allow."""
    k = _ITREE_THREAD_ELEMS // max(1, vec * loads) if want is None else want
    k = max(1, 1 << (max(k, 1).bit_length() - 1))  # floor to a power of two
    return math.gcd(k, max(wpr, 1))


def _loads_per_warp(remaining: int, lanes: int) -> int:
    # Loads a warp takes to cover its share, rounded UP to a power of two: the streaming carry
    # merges on the trailing-zero count of (load + 1), which only spans the tree when the load
    # count is a power of two.
    lpw = -(-remaining // lanes)
    return 1 << (lpw - 1).bit_length() if lpw > 1 else lpw


def itree_plan(
    N: int,
    M: int,
    itemsize: int,
    kchunk: int | None = None,
    vmul: int | None = None,
    vec_linear: bool = False,
    stage: bool | None = None,
):
    """The order's plan for this shape, or None when it does not apply.

    Mirrors upstream's try_inner_tree_reduction selection (inner_tree_plan.py owns the
    parameter computation, so the two cannot drift). None means "use the default order" --
    never "decline the call".
    """
    from .inner_tree_plan import (
        _K_MULTIROW_MAX_LOADS,
        _K_TWO_KERNEL_THRESHOLD,
        _next_power_of_2,
        compute_inner_tree_params,
    )

    if N < 1:
        return None
    kc = kchunk
    # NOT tile.vec_size: the order defines its vec from the itemsize alone and identity-pads a
    # ragged row, because the gcd form would make the DAG depend on N's divisibility.
    base_vec = 16 // itemsize
    vm = _ITREE_VEC_MUL if vmul is None else vmul
    vec = base_vec * vm
    wle = WARP * vec
    if N <= base_vec * _K_MULTIROW_MAX_LOADS:
        # The whole row lives in one thread's fragment, padded up to a power-of-two load count.
        # Always the BASE vec: this shape has no lane merge to trade away, so widening it would
        # move the bits for nothing.
        loads = _next_power_of_2(-(-N // base_vec))
        tm = tile.TileMap(N, itemsize, 1, loads, vec=base_vec)
        return _ItreePlan(
            "multirow",
            vec,
            0,
            _MULTIROW_ROWS_PER_BLOCK,
            _MULTIROW_MAX_DEPTH,
            ((0, N, loads, N),),
            (tm,),
        )
    prm = compute_inner_tree_params(N, M, vec)
    wpr = prm.num_warps
    if prm.num_batches > _K_TWO_KERNEL_THRESHOLD:
        # One tile serves both widths: it is the FULL batch's, and the short last batch reaches
        # fewer of its loads through a runtime per-warp bound (see tile._fold_itree).
        last = N - (prm.num_batches - 1) * prm.batch_total_elements
        chunk_full = prm.effective_loads * wle
        chunk_last = _loads_per_warp(last, wpr * wle) * wle
        tm = tile.TileMap(
            N,
            itemsize,
            WARP * wpr,
            prm.effective_loads,
            warp_major=True,
            vec=vec,
            exact=False,
        )
        k = _fuse_factor(kc, wpr, vec, prm.effective_loads)
        return _ItreePlan(
            "split",
            vec,
            wpr,
            1,
            prm.depth,
            ((0, prm.batch_total_elements, prm.effective_loads, chunk_full),),
            (tm,),
            (
                prm.num_batches,
                prm.batch_total_elements,
                last,
                chunk_full,
                chunk_last,
            ),
            k,
            vec_linear,
        )
    batches = []
    for b in range(prm.num_batches):
        off = b * prm.batch_total_elements
        remaining = min(prm.batch_total_elements, N - off)
        lpw = _loads_per_warp(remaining, wpr * wle)
        batches.append((off, remaining, lpw, lpw * wle))
    # EXACT per batch: the tile spans [off, off + wpr*lpw*wle), so when that stays inside the row
    # every load is unconditionally in range and the fold needs no per-element mask (and `load`
    # can take its unconditional wide path). A ragged tail or a short last batch is not exact.
    tms = tuple(
        tile.TileMap(
            N,
            itemsize,
            WARP * wpr,
            lpw_b,
            warp_major=True,
            vec=vec,
            exact=off_b + wpr * lpw_b * WARP * vec <= N,
        )
        for (off_b, _rem, lpw_b, _wc) in batches
    )
    k = _fuse_factor(kc, wpr, vec, prm.effective_loads)
    rpb = max(1, min(M, _ITREE_BLOCK_THREADS // max(1, WARP * (wpr // k))))
    # SMEM STAGING serves the SINGLE-BATCH shapes only for now, so the batch covers the row and
    # `hi` is one compile-time bound.
    span = wpr * prm.effective_loads * WARP * vec
    # `stage_e` is the per-lane run PER TILE, so smem stays at (stage_e + vec) * WARP per row no
    # matter how wide the batch is; the batch is covered in span/(stage_e*WARP) tiles, each ending
    # in one butterfly. Capped at the measured sweet spot rather than span/WARP.
    e = min(span // WARP, _ITREE_STAGE_E)
    while e > vec and (span // (e * WARP)) * e * WARP != span:
        e //= 2
    # Stage when the FULL per-lane run is achievable (a shorter one measured badly: 74.0 vs 51.3 at
    # span=512, E=16) AND there is more than one butterfly to remove -- at span == wle there is
    # already just one, so staging would buy nothing and still pay the smem round trip.
    want_stage = (e == _ITREE_STAGE_E and span > WARP * vec) if stage is None else stage
    if (
        want_stage
        and prm.num_batches == 1
        and not vec_linear
        # cp.async's 128-bit atom needs a statically 16-byte-aligned source, and the input wrap
        # can only declare that when vec divides N (align_bytes is gcd-based). A ragged row keeps
        # the register fold rather than silently narrowing the copy.
        and N % vec == 0
        and e % vec == 0
        and e <= tile.MAX_UNROLL
        and (e // vec) & (e // vec - 1) == 0
    ):
        return _ItreePlan(
            "looped",
            vec,
            wpr,
            min(M, prm.rows_per_block) or 1,
            prm.depth,
            tuple(batches),
            tms,
            (),
            1,
            False,
            e,
        )
    return _ItreePlan(
        "looped",
        vec,
        wpr,
        rpb,
        prm.depth,
        tuple(batches),
        tms,
        (),
        k,
        vec_linear,
    )


def itree_combine_plan(itree: _ItreePlan) -> _ItreePlan:
    """Stage 2 of the split shape: one thread per row, folding that row's partials."""
    return _ItreePlan(
        "combine",
        1,
        0,
        _MULTIROW_ROWS_PER_BLOCK,
        itree.depth,
        (),
        (),
        itree.split,
    )


class _RowConfig(NamedTuple):
    # Row-reduce knob set. reduce_row_tile() knobs left None are filled from row_config;
    # explicit values (autotuner / per-machine retune) override per field. This is the row
    # kernel's OWN config (xcta/K0 have differently-shaped knobs).
    tpr: int  # threads per row
    nt: int  # threads per block


def row_config(N: int, dtype_width: int, nfields: int = 1, hw=None) -> "_RowConfig":
    # Occupancy config from (N, dtype), with hw-scaled thresholds. tpr from the ladder,
    # nt from the size gate; the ladder's N-limits are proxies for "how wide before a row
    # needs more threads", which tracks smem capacity, so scaling them by hw.smem_scale
    # keeps the rule correct on other GPUs (1.0 on B200 -> the anchors reproduce exactly).
    #
    # nfields is ACCEPTED (signature parity + autotuner key) but deliberately NOT acted on: a
    # wide-accumulator trait does prefer a different config, but the fp32 and bf16 optima move in
    # OPPOSITE directions, so no single scalar rule serves both. Left to the autotuner, which
    # measures per key.
    # The ragged-N correctness clamp (tpr -> nt when the row tile isn't a clean vec*tpr
    # multiple) is applied at USE in RowReduce, not here -- it is a correctness invariant.
    smem = 1.0 if hw is None else hw.smem_scale
    # Wide-row rung first (byte-based, dtype-correct): a >=16KB row saturates 256
    # threads/row + a 256 block regardless of dtype. This overrides the element ladder,
    # which under-threads mid-N wide rows (e.g. fp32 N=4096-8192 -> tpr64/128, ~1.3x slow).
    if N * (dtype_width // 8) >= _WIDE_ROW_BYTES * smem:
        return _RowConfig(tpr=_TPR_MAX, nt=_NT_LARGE)
    tpr = next((t for limit, t in _TPR_LADDER if N <= limit * smem), _TPR_MAX)
    nt = _NT_SMALL if N <= _NT_GATE_N * smem else _NT_LARGE
    return _RowConfig(tpr=tpr, nt=nt)


def single_row_config(N: int, dtype_width: int, nfields: int = 1, hw=None):
    # Occupancy override for a ONE-ROW launch (the reduce-all one-shot), or None to leave the
    # ladder's pick standing. The ladder's small tpr exists so nt//tpr ROWS pack per block; with one
    # row there is nothing to pack and the GPU runs a fraction of one CTA. So give that row the
    # widest LADDER RUNG it can feed, one row per block.
    #
    # From _TPR_RUNGS, not a computed width: tpr is both the reduce-tree width and the block size, so
    # it must be a power of two AND a warp multiple -- a computed width silently returned a wrong
    # variance. MEASURED (M=1, bf16, vs ATen): var_mean 0.53-0.93x -> 1.47-1.62x at N=1024..4096,
    # with max.dim, aminmax and sum matching or improving from N=64 to 16384.
    cfg = row_config(N, dtype_width, nfields, hw)
    vec = math.gcd(N, 128 // dtype_width)
    feedable = min(_TPR_MAX, N // max(1, vec))  # vector loads this row can issue
    rungs = [t for t in _TPR_RUNGS if WARP <= t <= feedable]
    if not rungs or rungs[-1] <= cfg.tpr:
        return None
    return _RowConfig(tpr=rungs[-1], nt=rungs[-1])


def _launch_itree(trait, trait_key, plan, dt, wrap, N, tag, nouts=1, dsts=()):
    """Compile-or-fetch and launch one stage of the order. `wrap` rewraps per call."""
    op = tile.TileReduce(
        trait,
        dt,
        "row",
        N,
        nouts=nouts,
        final=plan.shape != "split",
        order="inner_tree",
        itree=plan,
    )

    # N is baked into the DAG, so only M rides in dynamically; the col axis's args and the
    # general axis's decode are None, not dummies (an unused Int32 param costs real time).
    def _args():
        mIns, mOuts = wrap()
        return (
            mIns,
            mOuts,
            Int32(N // plan.vec),
            None,
            Int32(N),
            None,
            None,
            None,
            None,
            None,
            None,
            _stream(),
        )

    # dsts: the kernel bakes each destination's element type, so two calls differing only in an
    # output dtype are different kernels. Every sibling driver keys on them; without it the second
    # call fetches the first's plan and the launch fails on a mismatched tensor.
    key = (tag, trait_key, dt, tuple(dsts)) + op.cache_sig
    build = lambda: _compile(op, *_args())  # noqa: E731
    cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")(*_args())


def _run_itree(trait, trait_key, x, out_dtypes, itree, nouts=1):
    """Run the inner-tree order for `x`, one launch per stage of its shape.

    Serves any trait: `nouts` projected outputs at the end, and the split shape's intermediate
    partials get one buffer PER TRAIT FIELD (an index or Welford accumulator is not one number).
    """
    M, N = x.shape
    dt = _L.torch2cute[x.dtype]
    # A ragged row's stride is not a vec multiple, so the wide load's alignment is only what the
    # gcd allows -- declaring 16 there would be a lie and the load faults.
    align = tile.align_bytes(N, x.element_size())
    wrap_in = lambda: _L.cute_tensor_dynM(  # noqa: E731
        x, align=align, ndim=2, read_only=True
    )
    if itree.shape != "split":
        outs = [torch.empty(M, device=x.device, dtype=d) for d in out_dtypes[:nouts]]
        wrap = lambda: (  # noqa: E731
            [wrap_in()],
            [_L.cute_tensor_dynM(o, ndim=1) for o in outs],
        )
        _launch_itree(
            trait,
            trait_key,
            itree,
            dt,
            wrap,
            N,
            "rowitree",
            nouts,
            tuple(o.dtype for o in outs),
        )
        return tuple(outs)
    # The split shape cannot bake its batch count, so it writes one partial per (row, batch) and
    # a second stage folds each row's partials LINEARLY -- both halves of upstream's two-kernel
    # path. Partials stay in the FIELD dtypes so the cross-batch fold rounds once.
    nbatch = itree.split[0]
    parts = [
        torch.empty(M * nbatch, device=x.device, dtype=_L.cute2torch[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    wrap1 = lambda: (  # noqa: E731
        [wrap_in()],
        [_L.cute_tensor_dynM(p, ndim=1) for p in parts],
    )
    _launch_itree(
        trait,
        trait_key,
        itree,
        dt,
        wrap1,
        N,
        "rowitree1",
        nouts,
        tuple(p.dtype for p in parts),
    )
    outs = [torch.empty(M, device=x.device, dtype=d) for d in out_dtypes[:nouts]]
    wrap2 = lambda: (  # noqa: E731
        [_L.cute_tensor_dynM(p, ndim=1, read_only=True) for p in parts],
        [_L.cute_tensor_dynM(o, ndim=1) for o in outs],
    )
    _launch_itree(
        trait,
        trait_key,
        itree_combine_plan(itree),
        dt,
        wrap2,
        N,
        "rowitree2",
        nouts,
        tuple(o.dtype for o in outs),
    )
    return tuple(outs)


def reduce_row_tile(
    trait,
    trait_key,
    x,
    out_dtypes,
    nouts=1,
    tpr=None,
    nt=None,
    final=True,
    unroll=None,
    use_tma=None,
    order=None,
):
    """Tile-based row reduction: reduce the contiguous last dim of a 2D `x` -> (M,).

    Returns a tuple of `nouts` outputs. tpr=1 is the NARROW-row shape (one thread per row);
    it stages through TMA where that wins, unless `use_tma` says otherwise. `order` selects the
    fold order: the default rolled one, or the reproducible inner-tree DAG when its env gate is
    set (see itree_plan, which covers every N of a plain 1-field reduction).
    """
    if x.dim() != 2 or not x.is_cuda or x.stride(-1) != 1:
        raise AssertionError(f"want 2D contiguous-last-dim CUDA, got {tuple(x.shape)}")
    M, N = x.shape
    # The reproducible-DAG order, when asked for: every N and every TRAIT, since the fold is
    # written on `leaf`/`combine` rather than on the serial `reduce`. Two things it still cannot
    # serve -- a raw stage-1 partial pass (`final=False`), whose consumer imposes its own layout,
    # and an explicit tpr, which is a launch-shape request a fixed DAG cannot honour. Those keep
    # the default order; neither falls back to aten.
    if order not in (None, "linear", "inner_tree"):
        raise ValueError(f"order must be None, 'linear' or 'inner_tree', got {order!r}")
    itree = None
    if (order == "inner_tree" or (order is None and inner_tree_order_enabled())) and (
        final and tpr is None
    ):
        itree = itree_plan(N, M, x.element_size())
    if order == "inner_tree" and itree is None:
        # An EXPLICIT request for a reproducible DAG must not be served with another one. Only
        # order=None (the env gate) may fall back to the default order, since that reads as "use
        # the order where it applies" rather than as a demand.
        raise ValueError(
            f"order='inner_tree' cannot be honoured here: {final=} {tpr=} "
            f"plan={itree_plan(N, M, x.element_size()) is not None}"
        )
    if itree is not None:
        return _run_itree(trait, trait_key, x, out_dtypes, itree, nouts)
    cfg = row_config(N, x.element_size() * 8, trait.nfields)
    # Unroll depth of the rolled wave loop. A SCALAR row (vec == 1, i.e. an odd or prime N,
    # where gcd(N, 16//itemsize) is 1) has no wide load to hide latency behind, so it wants more
    # independent loads in flight; a vectorized row does not, and a deeper unroll costs it.
    # MEASURED (tile us at unroll 4/8/16/32): vec=1 (512,4099) 3.6/3.6/3.5/5.1, (512,6143)
    # 4.1/4.0/4.8/6.4, (1,12289) 4.4/3.6/3.2/5.9; vec=4 (512,4096) 2.5/2.8/2.8/2.9.
    if unroll is None:
        unroll = 16 if tile.vec_size(N, x.element_size()) == 1 else 4
    tpr = max(WARP, cfg.tpr) if tpr is None else tpr
    nt = max(tpr, cfg.nt) if nt is None else nt
    nt -= nt % tpr  # rows_per_block must be whole
    if use_tma is None:
        use_tma = tpr == 1 and tma_ok(N, x.element_size(), M, x.device)
    dt = _L.torch2cute[x.dtype]
    op = tile.TileReduce(
        trait,
        dt,
        "row",
        N,
        tpr=tpr,
        nt=nt,
        nouts=nouts,
        final=final,
        unroll=unroll,
        use_tma=use_tma,
    )

    # final -> nouts projected results; stage 1 -> one RAW partial buffer per trait field
    ndst = nouts if final else trait.nfields
    outs = [torch.empty(M, device=x.device, dtype=dt) for dt in out_dtypes[:ndst]]
    nchunks = Int32(N // op.vec)
    nwaves = Int32(math.ceil((N // op.vec) / tpr))
    # Declared alignment is what lets the load stage emit the wide instruction; tile owns the
    # derivation so it cannot be forgotten here (it was, and cost 3x). The rolled paths take N
    # at RUNTIME, so they wrap with BOTH extents dynamic (N carrying divisibility=vec so the
    # wide load survives) and one compiled kernel serves the whole vec class; the TMA box
    # shape is compile-time, so that variant bakes N and only M stays dynamic.
    isz = x.element_size()
    align = op.tilemap.align_bytes(isz) if use_tma else tile.align_bytes(N, isz)

    def _wrap():
        mX = (
            _L.cute_tensor_dynM(x, align=align, ndim=2, read_only=True)
            if use_tma
            else _L.cute_tensor_dynMN(x, op.vec, align=align, read_only=True)
        )
        # q/npar belong to the COL axis: None, not a dummy value -- an unused Int32 param
        # costs real time (see tile.TileReduce.kernel).
        return (
            [mX],
            [_L.cute_tensor_dynM(o, ndim=1) for o in outs],
            nchunks,
            nwaves,
            Int32(N),
            None,  # q, npar: the col axis's split
            None,
            None,  # rvals, kvals, in_base, limit: the general axis's decode
            None,
            None,
            None,
            _stream(),
        )

    key = ("rowtile", trait_key, x.dtype, tuple(out_dtypes[:ndst])) + op.cache_sig
    build = lambda: _compile(op, *_wrap())  # noqa: E731
    fn = cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")
    fn(*_wrap())
    return tuple(outs)
