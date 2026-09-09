# ROW reductions: the launch policy for tile.TileReduce on the row axis. The body is in
# tile.py; this module owns the measured launch shapes, the narrow-row gates and the plan
# cache. The chunk loop is ROLLED, so one compiled kernel covers every N in a vec class.

import math
from typing import NamedTuple

from cutlass import Int32

import torch

from ...cutedsl.dtypes import torch2cute
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
):
    """Tile-based row reduction: reduce the contiguous last dim of a 2D `x` -> (M,).

    Returns a tuple of `nouts` outputs. tpr=1 is the NARROW-row shape, TMA-staged where that
    wins. `order` selects the fold order; see itree_plan for the reproducible one.
    """
    if x.dim() != 2 or not x.is_cuda or x.stride(-1) != 1:
        raise AssertionError(f"want 2D contiguous-last-dim CUDA, got {tuple(x.shape)}")
    M, N = x.shape
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
        use_tma = tpr == 1 and tma_ok(N, x.element_size(), M, x.device)
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
    align = op.tilemap.align_bytes(isz) if use_tma else tile.align_bytes(N, isz)

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

    key = ("rowtile", trait_key, x.dtype, tuple(out_dtypes[:ndst])) + op.cache_sig
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
