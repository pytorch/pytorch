# ROW reductions: the launch policy for tile.TileReduce on the row axis. The body is in
# tile.py; this module owns the measured launch shapes, the narrow-row gates and the plan
# cache. The chunk loop is ROLLED, so one compiled kernel covers every N in a vec class.

import math
from typing import NamedTuple

from cutlass import Int32

import torch

from ...cutedsl.dtypes import cute2torch, torch2cute
from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan
from .._cutedsl.traits import WARP
from . import tile


_compile = _L.compile_kernel
_stream = _L.stream
_CACHE = {}


# --- Row-reduce occupancy heuristic, as DATA (see row_config) --- threads-per-row ladder,
# first matching row wins. Small N takes one warp per row so many rows pack per block with no
# cross-warp reduce. The N-limits are B200 anchors that row_config scales by hw.
_TPR_LADDER = ((64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128))
_TPR_MAX = 256
# Every legal tpr, widest last. Powers of two: tpr sets the cross-thread reduce width and,
# when it doubles as the block size, the warp count -- both need one.
_TPR_RUNGS = tuple(t for _, t in _TPR_LADDER) + (_TPR_MAX,)
# threads-per-block (nt) gate: small rows use a smaller block, wider rows the larger.
_NT_SMALL, _NT_LARGE, _NT_GATE_N = 128, 256, 16 * 1024
# Wide-row rung: past 16 KB a row needs the full 256 threads, which the dtype-blind element
# ladder under-threads (1.1-1.4x). In BYTES, so it is dtype-correct with no per-dtype table.
_WIDE_ROW_BYTES = 16 * 1024


# --- NARROW rows: tpr == 1 --- `tpr` floors at a WARP wherever lanes are merged, so a narrow
# row leaves most of each warp idle (the packed shape measured 4.0x slower at (1048576, 32)).
# tpr == 1 merges nothing, so it serves any trait. The width ceiling is derived from
# MAX_UNROLL, since the whole row is one thread's unroll, and sits far above the crossover.
_MAX_NARROW_N = min(256, tile.MAX_UNROLL)
# MEASURED ladder of (minimum rows, per-thread chunk budget): one thread per row shrinks the
# grid ~tpr times, so it needs enough rows to fill the SMs, and more of them the wider the
# row. In vec-CHUNKS so it carries across dtypes. Tiered because one bound cannot serve both
# ends -- 1.14-1.87x at M=4096, up to 33.7x at M=262144.
_CHUNK_LADDER = ((65536, 32), (16384, 16), (4096, 6))

# TMA-STAGED LOAD, for the one regime the direct load cannot reach SOL: thread t reads row t,
# so the lane stride is a whole row and the direct load only holds 91-93% of peak while two
# lanes share a 128-byte line (7001 GB/s at N=16 against 4584 at N=32). It is OVER-FETCH, so
# the fix is a contiguous access, which a TMA box is. Worth 1.49-1.86x, but ONLY with the smem
# rotation -- without it a regression -- so it is gated to the po2 fp32 N that mask assumes.
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
        # Through the memoized caps: this is evaluated on EVERY launch of the band, ahead of the
        # plan-cache lookup, and the raw device query costs ~1.3us.
        from .._cutedsl import hw_caps as _hw

        if _hw.caps(device).cc[0] < 9:
            return False  # TMA is sm_90+
    return True


# --- INNER-TREE ORDER (opt-in) --- Every other order derives its add association from the
# LAUNCH SHAPE, so it moves with tpr, nt or M. This one fixes the DAG from N alone, which is
# what makes it hash-pinnable and so usable for a determinism claim. It must cover EVERY N,
# since a shape it skipped would silently keep the launch-shape order.
#
# OPT-IN because it costs 0.92-1.41x the rolled fold's device time and gives up the N-free
# compile key. Its env var is ours, not upstream's: upstream's kernels register first and keep
# their eligible calls, so the two must be switchable independently.
_INNER_TREE_ENV = "PYTORCH_NATIVE_INNER_TREE"
# Block size for the two ONE-THREAD-PER-ROW shapes (upstream's kMultiRowThreads and
# kAccumulateThreads), and the multirow carry's cap, which is fixed rather than plan-derived.
_MULTIROW_ROWS_PER_BLOCK = 128
_MULTIROW_MAX_DEPTH = 6

# Per-thread width for the order, in ELEMENTS. Width is bought by fusing adjacent CHUNKS,
# which keeps a full warp on each and so keeps the loads coalesced: worth 1.79x at
# (65536, 1024). 64 is within 2% of the best point everywhere measured, and 256 falls off a
# cliff (127.6us against 56.7).
_ITREE_THREAD_ELEMS = 64

# Vector-width multiplier for the order's `vec`. 1: a wider thread run strides the lanes by
# its width, ~2x per doubling. Bit-exact at any value -- `bte` is what fixes the bits.
_ITREE_VEC_MUL = 1

# Target BLOCK SIZE, which is how rows-per-block is picked. DAG-free, so pure occupancy. 256,
# because a 64-thread block starves the SM of rows at small N (145.3 against 68.2us).
_ITREE_BLOCK_THREADS = 256

# SMEM-STAGED FOLD, for NARROW per-lane runs. A coalesced load leaves a lane owning only `vec`
# columns while in-register tree levels want a contiguous run; staging breaks the tie, so one
# butterfly replaces many. Bit-neutral. cp.async in TILES, which closes the mid-band's 1.28x
# deficit (48.2 -> 40.9us at (65536, 1024)). Gated on a single batch and the FULL per-lane run:
# a shorter run, a register-staged copy and an untiled buffer each measured worse.

# Columns per lane PER TILE. 32 is where the untiled sweep bottomed out (40.7us at N=1024); tiling
# holds every wider batch at that same per-lane run and the same 4.6 KB of smem per row.
_ITREE_STAGE_E = 32


def inner_tree_order_enabled() -> bool:
    """Is the reproducible-DAG order requested? Read live, so tests can toggle it."""
    import os

    return os.environ.get(_INNER_TREE_ENV, "") not in ("", "0")


class _ItreePlan(NamedTuple):
    """Everything the order's DAG depends on, all compile-time: one of three SHAPES, chosen from
    N as upstream's host dispatch chooses between its three kernels.

    The shape is part of the DAG, so it cannot be a launch-time preference. The split shape's
    batch index is a RUNTIME value, so its two entries are the two distinct WIDTHS.
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
    # unchanged, and the k results combine exactly where the cross-chunk merge would have.
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
    # merges on the trailing-zero count of (load + 1), which only spans the tree at a power of two.
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
    """The order's plan for this shape, or None when it does not apply -- which means "use the
    default order", never "decline the call". Mirrors upstream's selection.
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
        # The whole row lives in one thread's fragment, padded to a power-of-two load count. Always
        # the BASE vec: with no lane merge to trade away, widening would move the bits for nothing.
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
    # EXACT per batch: when the tile stays inside the row every load is unconditionally in range,
    # so the fold needs no per-element mask. A ragged tail or a short last batch is not exact.
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
    # SMEM STAGING serves the SINGLE-BATCH shapes only: it needs the batch to cover the row, so the
    # bound is compile-time, and the per-lane run within the unroll ceiling. Only worth it when it
    # REMOVES butterflies -- at one already there is nothing to win and the round trip still costs.
    span = wpr * prm.effective_loads * WARP * vec
    # `stage_e` is the per-lane run PER TILE, so smem stays fixed however wide the batch is and
    # the batch is covered in as many tiles as that takes, each ending in one butterfly.
    e = min(span // WARP, _ITREE_STAGE_E)
    while e > vec and (span // (e * WARP)) * e * WARP != span:
        e //= 2
    # Stage only for the FULL per-lane run (a shorter one measured 74.0 against 51.3us) and only
    # with more than one butterfly to remove, or the smem round trip buys nothing.
    want_stage = (e == _ITREE_STAGE_E and span > WARP * vec) if stage is None else stage
    if (
        want_stage
        and prm.num_batches == 1
        and not vec_linear
        # cp.async's 128-bit atom needs a statically 16-byte-aligned source, which the wrap can only
        # declare when vec divides N. A ragged row keeps the register fold rather than narrowing.
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
    # Row-reduce knob set: knobs left None are filled from row_config, and explicit values
    # override per field. This is the row kernel's OWN config -- the other axes differ in shape.
    tpr: int  # threads per row
    nt: int  # threads per block


def row_config(N: int, dtype_width: int) -> "_RowConfig":
    # Occupancy config from (N, dtype). The ladder's N-limits are proxies for how wide a row gets
    # before it needs more threads. No nfields term: the fp32 and bf16 optima move in OPPOSITE
    # directions, so no scalar rule serves both.
    #
    # Wide-row rung first, and byte-based: a >=16KB row saturates 256 threads whatever the dtype.
    # This overrides the element ladder, which under-threads mid-N wide rows by ~1.3x.
    if N * (dtype_width // 8) >= _WIDE_ROW_BYTES:
        return _RowConfig(tpr=_TPR_MAX, nt=_NT_LARGE)
    tpr = next((t for limit, t in _TPR_LADDER if N <= limit), _TPR_MAX)
    nt = _NT_SMALL if N <= _NT_GATE_N else _NT_LARGE
    return _RowConfig(tpr=tpr, nt=nt)


def single_row_config(N: int, dtype_width: int):
    # Occupancy override for a ONE-ROW launch, or None to leave the ladder's pick standing. The
    # ladder's small tpr exists so rows pack per block; with one row the GPU runs a fraction of
    # one CTA, so give that row the widest rung it can feed. From _TPR_RUNGS rather than a
    # computed width, since tpr is both tree width and block size -- a computed one returned a
    # wrong variance. Measured 0.53-0.93x -> 1.47-1.62x of ATen on var_mean.
    cfg = row_config(N, dtype_width)
    vec = math.gcd(N, 128 // dtype_width)
    feedable = min(_TPR_MAX, N // max(1, vec))  # vector loads this row can issue
    rungs = [t for t in _TPR_RUNGS if WARP <= t <= feedable]
    if not rungs or rungs[-1] <= cfg.tpr:
        return None
    return _RowConfig(tpr=rungs[-1], nt=rungs[-1])


def _launch_itree(
    trait, trait_key, plan, dt, fakes, operands, N, tag, nouts=1, dsts=()
):
    """Compile-or-fetch and launch one stage of the order.

    `align` is part of the key: the declared alignment is baked into the kernel, so a call
    whose pointer meets less must not inherit a wider claim.
    """
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
    def _args(pair):
        mIns, mOuts = pair
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
            None,
            None,
            _stream(),
        )

    # The kernel bakes each destination's element type, so two calls differing only in an output
    # dtype are different kernels -- without this the second fetches the first's plan and fails.
    key = (tag, trait_key, dt, tuple(dsts)) + op.cache_sig
    build = lambda: _compile(op, *_args(fakes))  # noqa: E731
    cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")(*_args(operands))


def _run_itree(trait, trait_key, x, out_dtypes, itree, nouts=1):
    """Run the inner-tree order for `x`, one launch per stage of its shape.

    Serves any trait: the split shape's partials get one buffer PER TRAIT FIELD. `out` names
    the result tensors, which must be 1-D unit-stride.
    """
    M, N = x.shape
    dt = torch2cute[x.dtype]
    # A ragged row's stride is not a vec multiple, so declaring 16 would be a lie and the load
    # faults; so would a compact input at a non-zero STORAGE OFFSET, whose strides are fine but
    # whose base pointer is not. Declare what the pointer meets and key on it, since cache_sig
    # has no alignment field of its own.
    # What N allows, narrowed to what the base pointer meets: a wider claim than the pointer
    # honours is rejected at launch, and N alone cannot see a storage offset.
    align = _declared_align(x, tile.align_bytes(N, x.element_size()))
    # N is baked into the DAG, so the row extent is static and only M rides in dynamically.
    fake_in = _L.fake_compact(dt, (_L.sym(), N), order=(1, 0), align=align)
    fake_1d = lambda t: _L.fake_compact(  # noqa: E731
        torch2cute[t.dtype], (_L.sym(),)
    )
    if itree.shape != "split":
        outs = [torch.empty(M, device=x.device, dtype=d) for d in out_dtypes[:nouts]]
        _launch_itree(
            trait,
            trait_key,
            itree,
            dt,
            ([fake_in], [fake_1d(o) for o in outs]),
            ([_L.read_only(x)], list(outs)),
            N,
            "rowitree",
            nouts,
            tuple(o.dtype for o in outs),
        )
        return tuple(outs)
    # The split shape cannot bake its batch count, so it writes one partial per (row, batch) and a
    # second stage folds them LINEARLY. Partials stay in the FIELD dtypes, so rounding happens once.
    nbatch = itree.split[0]
    parts = [
        torch.empty(M * nbatch, device=x.device, dtype=cute2torch[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    _launch_itree(
        trait,
        trait_key,
        itree,
        dt,
        ([fake_in], [fake_1d(p) for p in parts]),
        ([_L.read_only(x)], list(parts)),
        N,
        "rowitree1",
        nouts,
        tuple(p.dtype for p in parts),
    )
    outs = [torch.empty(M, device=x.device, dtype=d) for d in out_dtypes[:nouts]]
    _launch_itree(
        trait,
        trait_key,
        itree_combine_plan(itree),
        dt,
        ([fake_1d(p) for p in parts], [fake_1d(o) for o in outs]),
        ([_L.read_only(p) for p in parts], list(outs)),
        N,
        "rowitree2",
        nouts,
        tuple(o.dtype for o in outs),
    )
    return tuple(outs)


def _declared_align(x, natural: int) -> int:
    """The alignment the wrap may DECLARE for `x`: what N allows, narrowed to what its base
    pointer meets. Both are powers of two, so halving terminates at the element width.
    """
    # const_data_ptr, so reading the address does not materialize a COW tensor.
    with torch._C.DisableTorchFunctionSubclass():
        ptr = x.const_data_ptr()
    align = natural
    while align > x.element_size() and ptr % align:
        align //= 2
    return align


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

    Returns a tuple of `nouts` outputs. tpr=1 is the NARROW-row shape, TMA-staged where that
    wins. `order` selects the fold order; see itree_plan for the reproducible one.
    """
    if x.dim() != 2 or not x.is_cuda or x.stride(-1) != 1:
        raise AssertionError(f"want 2D contiguous-last-dim CUDA, got {tuple(x.shape)}")
    M, N = x.shape
    # The reproducible-DAG order, for every N and every TRAIT, since the fold is written on
    # leaf/combine rather than the serial reduce. Two things it cannot serve: a raw stage-1
    # partial pass, whose consumer imposes a layout, and an explicit tpr, which is a launch-shape
    # request a fixed DAG cannot honour. Both keep the default order; neither falls back to aten.
    if order not in (None, "linear", "inner_tree"):
        raise ValueError(f"order must be None, 'linear' or 'inner_tree', got {order!r}")
    itree = None
    if (order == "inner_tree" or (order is None and inner_tree_order_enabled())) and (
        final and tpr is None
    ):
        itree = itree_plan(N, M, x.element_size())
    if order == "inner_tree" and itree is None:
        # An EXPLICIT request for a reproducible DAG must not be served with a different one. Only
        # order=None -- the env gate, which reads as "where it applies" -- may fall back.
        raise ValueError(
            f"order='inner_tree' cannot be honoured here: {final=} {tpr=} "
            f"plan={itree_plan(N, M, x.element_size()) is not None}"
        )
    if itree is not None:
        return _run_itree(trait, trait_key, x, out_dtypes, itree, nouts)
    cfg = row_config(N, x.element_size() * 8)
    # Unroll depth of the rolled wave loop. A SCALAR row (an odd or prime N) has no wide load to
    # hide latency behind and wants more loads in flight; a vectorized row pays for the depth.
    # Measured across unroll 4/8/16/32, 4 is at or within noise of the best at every shape.
    if unroll is None:
        unroll = 16 if tile.vec_size(N, x.element_size()) == 1 else 4
    tpr = max(WARP, cfg.tpr) if tpr is None else tpr
    nt = max(tpr, cfg.nt) if nt is None else nt
    nt -= nt % tpr  # rows_per_block must be whole
    if use_tma is None:
        natural = tile.align_bytes(N, x.element_size())
        use_tma = (
            tpr == 1
            and _declared_align(x, natural) == natural
            and tma_ok(N, x.element_size(), M, x.device)
        )
    dt = torch2cute[x.dtype]
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
    # Declared alignment is what lets the load emit the wide instruction, and tile owns the
    # derivation so it cannot be forgotten here (it was, and cost 3x). The rolled paths take N at
    # RUNTIME, wrapping with both extents dynamic so one kernel serves a vec class; the TMA box
    # shape is compile-time, so that variant bakes N.
    isz = x.element_size()
    # Narrowed to what the base pointer meets; use_tma already required the natural claim.
    align = (
        op.tilemap.align_bytes(isz)
        if use_tma
        else _declared_align(x, tile.align_bytes(N, isz))
    )

    def _fake():
        # Compile-time descriptors: 2D row-major, both extents dynamic (the inner one divisible by
        # vec, so one kernel serves the vec class) EXCEPT under TMA, whose descriptor is static.
        # The col axis's args are None rather than dummies -- an unused Int32 param costs real time.
        inner = N if use_tma else _L.sym(op.vec)
        return (
            [_L.fake_compact(dt, (_L.sym(), inner), order=(1, 0), align=align)],
            [_L.fake_compact(torch2cute[o.dtype], (_L.sym(),)) for o in outs],
            nchunks,
            nwaves,
            Int32(N),
            None,  # q, npar: the col axis's split
            None,
            None,  # the general axis's decode: exts, strides, in_base, limit
            None,
            None,
            None,
            None,
            None,
            _stream(),
        )

    # align is part of the KEY now that it depends on the pointer: two calls of the same shape
    # can differ in it, and the declared value is baked into the kernel.
    dts = tuple(out_dtypes[:ndst])
    key = ("rowtile", trait_key, x.dtype, dts, align) + op.cache_sig
    build = lambda: _compile(op, *_fake())  # noqa: E731
    fn = cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")
    # The real operands: read_only on the INPUT, or a COW input materializes on export. The other
    # axes' args are None rather than dummies -- an unused Int32 param costs real time.
    fn(
        [_L.read_only(x)],
        list(outs),
        nchunks,
        nwaves,
        Int32(N),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        _stream(),
    )
    return tuple(outs)
