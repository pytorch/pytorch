# Row-reduction launch policy and plan cache for tile.TileReduce. Runtime loops share
# each kernel across a vector class; narrow rows may use one thread and TMA staging.
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


# First matching B200 threads-per-row anchor wins; small rows pack without cross-warp merge.
_TPR_LADDER = ((64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128))
_TPR_MAX = 256
# Legal power-of-two reduction/block widths, widest last.
_TPR_RUNGS = tuple(t for _, t in _TPR_LADDER) + (_TPR_MAX,)
# Small rows use fewer threads/block.
_NT_SMALL, _NT_LARGE, _NT_GATE_N = 128, 256, 16 * 1024
# Rows >=16 KB need 256 threads; the element ladder underthreads them by 1.1-1.4x.
_WIDE_ROW_BYTES = 16 * 1024


# Narrow rows: merged mappings floor tpr at one warp, wasting lanes and measuring 4.0x
# slower at (1048576, 32). tpr=1 serves any trait without merging; MAX_UNROLL bounds
# its whole-row unroll above the measured crossover.
_MAX_NARROW_N = min(256, tile.MAX_UNROLL)
# Measured (minimum rows, vector-chunk budget) ladder: tpr=1 shrinks the grid and
# needs more rows as rows widen. Tiers generalize across dtypes and measured 1.14-1.87x
# at M=4096, up to 33.7x at M=262144.
_CHUNK_LADDER = ((65536, 32), (16384, 16), (4096, 6))

# Direct tpr=1 loads over-fetch once adjacent rows no longer share a 128-byte line:
# 7001 GB/s at N=16 versus 4584 at N=32. TMA with smem rotation gains 1.49-1.86x;
# the rotation mask requires power-of-two fp32 N.
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
        # This runs before every plan lookup; memoize the ~1.3us device query.
        from .._cutedsl import hw_caps as _hw

        if _hw.caps(device).cc[0] < 9:
            return False  # TMA is sm_90+
    return True


# --- INNER-TREE ORDER (opt-in) --- Unlike launch-shape orders, this fixes the DAG from N,
# making it hash-pinnable, and covers every N to prevent silent fallback. It costs
# 0.92-1.41x the rolled fold and requires per-N compilation. Its independent env var avoids
# affecting upstream kernels, which register first.
_INNER_TREE_ENV = "PYTORCH_NATIVE_INNER_TREE"
# Block size for both one-thread-per-row shapes and fixed multirow carry cap.
_MULTIROW_ROWS_PER_BLOCK = 128
_MULTIROW_MAX_DEPTH = 6

# Per-thread elements from fusing adjacent chunks while keeping loads coalesced: 1.79x at
# (65536, 1024). 64 is within 2% of the measured optimum; 256 takes 127.6us versus 56.7us.
_ITREE_THREAD_ELEMS = 64

# Vector-width multiplier; each doubling costs ~2x. `bte`, not this, fixes the bits.
_ITREE_VEC_MUL = 1

# Occupancy-only block target; 64 threads starves small-N rows (145.3us versus 68.2us).
_ITREE_BLOCK_THREADS = 256

# Smem resolves coalesced vec loads versus contiguous per-lane trees, replacing butterflies
# without changing bits. cp.async tiles cut (65536, 1024) from 48.2 to 40.9us. Single-batch
# full runs only: shorter runs, register staging, and untiled buffers lost. 32 columns/lane
# reached 40.7us at N=1024 and caps wider rows at 4.6 KB smem each.
_ITREE_STAGE_E = 32


def inner_tree_order_enabled() -> bool:
    """Is the reproducible-DAG order requested? Read live, so tests can toggle it."""
    import os

    return os.environ.get(_INNER_TREE_ENV, "") not in ("", "0")


class _ItreePlan(NamedTuple):
    """Compile-time DAG state for one of upstream's three N-selected shapes.

    Shape is not a launch preference. Split's runtime batch index has two distinct widths.
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
    # Adjacent chunks fused without changing their trees or cross-chunk merge.
    kchunk: int = 1
    # Fold each thread run linearly instead of as a tree, changing the DAG.
    vec_linear: bool = False
    # Staged columns/lane; 0 disables. One butterfly preserves ATen's chunk nesting.
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
    # Round up so trailing-zero streaming carry spans a power-of-two tree.
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
    """Return the upstream-matching plan, or None to use default order without declining."""
    from .inner_tree_plan import (
        _K_MULTIROW_MAX_LOADS,
        _K_TWO_KERNEL_THRESHOLD,
        _next_power_of_2,
        compute_inner_tree_params,
    )

    if N < 1:
        return None
    kc = kchunk
    # Itemsize-only vec plus identity padding keeps the DAG independent of N divisibility.
    base_vec = 16 // itemsize
    vm = _ITREE_VEC_MUL if vmul is None else vmul
    vec = base_vec * vm
    wle = WARP * vec
    if N <= base_vec * _K_MULTIROW_MAX_LOADS:
        # One fragment, padded to power-of-two loads. Widening base vec would only move bits.
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
        # One full-batch tile serves both widths; runtime bounds shorten the last batch.
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
    # A batch wholly inside the row is exact; ragged or short final batches retain masks.
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
    # Stage one bounded batch only when removing multiple butterflies repays the smem trip.
    span = wpr * prm.effective_loads * WARP * vec
    # Fixed-smem tiles each end in one butterfly over stage_e columns/lane.
    e = min(span // WARP, _ITREE_STAGE_E)
    while e > vec and (span // (e * WARP)) * e * WARP != span:
        e //= 2
    # Require a full run (short: 74.0us versus 51.3us) and multiple butterflies.
    want_stage = (e == _ITREE_STAGE_E and span > WARP * vec) if stage is None else stage
    if (
        want_stage
        and prm.num_batches == 1
        and not vec_linear
        # 128-bit cp.async needs static 16-byte alignment; ragged rows keep register folding.
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
    # Row-kernel defaults; explicit arguments override them.
    tpr: int  # threads per row
    nt: int  # threads per block


def row_config(N: int, dtype_width: int) -> "_RowConfig":
    # Occupancy by N and dtype; no nfields rule fits opposing fp32/bf16 optima.
    # Check the byte rung first because the element ladder underthreads it by ~1.3x.
    if N * (dtype_width // 8) >= _WIDE_ROW_BYTES:
        return _RowConfig(tpr=_TPR_MAX, nt=_NT_LARGE)
    tpr = next((t for limit, t in _TPR_LADDER if N <= limit), _TPR_MAX)
    nt = _NT_SMALL if N <= _NT_GATE_N else _NT_LARGE
    return _RowConfig(tpr=tpr, nt=nt)


def single_row_config(N: int, dtype_width: int):
    # A lone row cannot exploit packing, so use its widest feedable legal rung. Computed
    # widths changed tree shape and returned wrong variance. Measured var_mean improves
    # from 0.53-0.93x to 1.47-1.62x of ATen.
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
    """Launch one stage, keying the baked alignment to prevent overstating later pointers."""
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

    # N is static and only M is dynamic; None omits costly unused axis arguments.
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

    # Destination types are baked, so include them to prevent a wrong cached plan.
    key = (tag, trait_key, dt, tuple(dsts)) + op.cache_sig
    build = lambda: _compile(op, *_args(fakes))  # noqa: E731
    cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")(*_args(operands))


def _run_itree(trait, trait_key, x, out_dtypes, itree, nouts=1):
    """Run one launch per stage; split shapes allocate one partial buffer per trait field."""
    M, N = x.shape
    dt = torch2cute[x.dtype]
    # Storage offsets may underalign; declare and key the pointer-supported width.
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
    # Split writes one field-typed partial per (row, batch), then folds them linearly.
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
    """Return the greatest N-allowed alignment met by `x`'s base pointer."""
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
    """Reduce 2-D `x` rows, returning outputs or raw field partials when `final=False`."""
    if x.dim() != 2 or not x.is_cuda or x.stride(-1) != 1:
        raise AssertionError(f"want 2D contiguous-last-dim CUDA, got {tuple(x.shape)}")
    M, N = x.shape
    # leaf/combine serves every trait and N. The opt-in gate leaves partial stages and
    # explicit launch shapes on the default order; explicit requests raise below.
    if order not in (None, "linear", "inner_tree"):
        raise ValueError(f"order must be None, 'linear' or 'inner_tree', got {order!r}")
    itree = None
    if (order == "inner_tree" or (order is None and inner_tree_order_enabled())) and (
        final and tpr is None
    ):
        itree = itree_plan(N, M, x.element_size())
    if order == "inner_tree" and itree is None:
        # Explicit inner_tree cannot silently use another DAG; only the env gate may fall back.
        raise ValueError(
            f"order='inner_tree' cannot be honoured here: {final=} {tpr=} "
            f"plan={itree_plan(N, M, x.element_size()) is not None}"
        )
    if itree is not None:
        return _run_itree(trait, trait_key, x, out_dtypes, itree, nouts)
    cfg = row_config(N, x.element_size() * 8)
    # Scalar rows use 16 to hide narrow-load latency; vectorized rows use 4, at or near
    # the measured optimum across 4/8/16/32.
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

    # Final projects nouts; stage 1 stores one raw buffer per field.
    ndst = nouts if final else trait.nfields
    outs = [torch.empty(M, device=x.device, dtype=dt) for dt in out_dtypes[:ndst]]
    nchunks = Int32(N // op.vec)
    nwaves = Int32(math.ceil((N // op.vec) / tpr))
    # Declare alignment to retain wide loads (worth 3x), narrowed for storage offsets
    # outside TMA. Runtime folds share a vector class; TMA bakes its box width.
    isz = x.element_size()
    align = (
        op.tilemap.align_bytes(isz)
        if use_tma
        else _declared_align(x, tile.align_bytes(N, isz))
    )

    def _fake():
        # TMA bakes N; runtime folds share a vector class. None omits unused column args.
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

    # Pointer-dependent alignment is compiled, so include it in the key.
    dts = tuple(out_dtypes[:ndst])
    key = ("rowtile", trait_key, x.dtype, dts, align) + op.cache_sig
    build = lambda: _compile(op, *_fake())  # noqa: E731
    fn = cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")
    # read_only avoids COW materialization; None omits unused-axis arguments.
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
