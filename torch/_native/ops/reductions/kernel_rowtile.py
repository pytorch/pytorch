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

    Returns a tuple of `nouts` outputs. tpr=1 is the NARROW-row shape (one thread per row);
    it stages through TMA where that wins, unless `use_tma` says otherwise.
    """
    if x.dim() != 2 or not x.is_cuda or x.stride(-1) != 1:
        raise AssertionError(f"want 2D contiguous-last-dim CUDA, got {tuple(x.shape)}")
    M, N = x.shape
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
