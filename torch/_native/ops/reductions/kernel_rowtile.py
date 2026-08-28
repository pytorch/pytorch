# ROW reductions: launch policy for tile.TileReduce on the row axis (reduce the contiguous
# last dim of a (M, N) input). The kernel body lives in tile.py -- this module owns the
# measured launch shapes, the narrow-row gates, and the plan cache.
#
# Covers 1 or 2 outputs and either a final projection or raw stage-1 partials (which is what
# makes it reusable as the cross-CTA driver's stage 1), with a ROLLED chunk loop so ONE
# compiled kernel serves every N in a vec class -- tpr and nt are compile-time while N itself
# arrives as a runtime arg.
#
# Two options serve NARROW rows, where packing threads onto a row wastes most of each warp:
# tpr=1 gives a row one thread, and past a 128-byte lane stride the load stages through a TMA
# box (see _TMA_MIN_STRIDE).

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
# `tpr` floors at one WARP whenever lanes have to be merged (warp_reduce shuffles across a
# full warp), so a row only `N // vec` vec-chunks wide leaves `WARP - chunks` lanes with
# nothing to load. Measured lane utilization, and what it cost the packed shape at large M:
#
#   N=16  ->   4 chunks, tpr 128 (the ragged guard forces 1 row/block) ->  3.1% util
#   N=32  ->   8 chunks, tpr  32                                       -> 25.0% util
#   N=64  ->  16 chunks, tpr  32                                       -> 50.0% util
#   N=128 ->  32 chunks, tpr  32                                       ->  100% util
#
# and 25.0% util lined up with the packed shape being 4.0x slower than tpr == 1 at
# (1048576, 32). At tpr == 1 there is no lane merge at all, so the shape is trait-agnostic:
# index traits and multi-field traits (Welford nfields=3) need no extra machinery. `nt` still
# comes from the ladder below (128 everywhere in this band), so the only knob is tpr.
#
# Upper bound on the row width tpr == 1 will take. The whole row lives in one thread, so the
# per-thread unroll IS N // vec; must not exceed tile.MAX_UNROLL -- derived from it rather
# than restated, so the two cannot drift. The measured crossover is far below either (see
# `narrow_row`); this is the safety ceiling, not the perf choice.
_MAX_NARROW_N = min(256, tile.MAX_UNROLL)
# MEASURED ladder of (minimum rows, per-thread chunk budget). A launch-shape pick among our
# own kernels, like the one-shot-vs-xcta choice -- not a capability gate on whether we serve
# the op.
#
# One thread per row makes the grid ~tpr times smaller than the packed shape's, so it needs
# enough rows to fill the SMs, and the wider the row the more rows it needs. Verified on B200
# across nfields 1/2/3 (sum, mean, amax, argmax, max.dim, var), device time, gate on vs off,
# one process per shape:
#
#   M       chunks   worst op     best op
#   4096         6   1.14x sum    1.87x var
#   8192         6   1.38x sum    2.81x var
#   16384       16   1.32x amax   1.60x var
#   65536       24   1.24x sum    2.95x var
#   65536       32   0.98x var    1.35x max.dim
#   1048576     8    4.23x amax   9.10x var
#   262144      4    9.25x sum   33.73x var
#   16384       32   0.68x var    1.03x sum   <- REGRESSES, excluded by the ladder
#
# The last row is why the budget is tiered rather than a single bound: at 32 chunks the
# multi-field traits lose at M=16384 but not at M=65536.
#
# Bounds are in vec-CHUNKS (N // vec), not raw N: chunks is what sets loads and registers per
# thread, so the bound carries across dtypes (32 chunks is 512 B/thread at fp32 and at fp16).
# Only fp32 was measured for perf; correctness is covered for fp16/bf16/fp64 too.
_CHUNK_LADDER = ((65536, 32), (16384, 16), (4096, 6))

# TMA-STAGED LOAD, for the one regime where the direct load is not already at SOL.
#
# MEASURED (512 MiB footprint, past L2): the direct tpr == 1 load runs at 91-93% of peak
# while >= 2 lanes of a warp share a 128-byte line, and collapses to ~60% the moment they do
# not. Thread t reads row t, so the stride between lanes is N*itemsize, and the cliff is at
# exactly N*itemsize >= 128 -- 7001 GB/s at N=16 (64 B stride) vs 4584 at N=32 (128 B).
#
# The problem is OVER-FETCH, not latency: each lane pulls its own line and uses part of it.
# So the fix is to make the gmem access contiguous, which a TMA box does by construction --
# a (rows, N) box spanning FULL rows is one contiguous gmem region, moved by a single
# descriptor-driven transfer with no per-lane addressing at all.
#
# DEPTH 1 on purpose. Multi-stage pipelining cannot un-fetch bytes: a prior experiment in
# this repo measured the depth curve flat on a bandwidth-bound reduction (D1 2491us vs D2
# 2437us, D3..D6 unchanged). Depth > 1 is also where PipelineTmaAsync deadlocks -- its empty
# barrier signals per warp-lane-0, so a full-block consumer group gives the wrong
# arrive_count on buffer reuse.
#
# MEASURED with the smem rotation (tile.fold_smem_rotated, without which this is a
# REGRESSION): 1.49-1.86x over the direct load, reaching 6943 GB/s = 90% of theoretical peak
# at (4194304, 32) -- matching the ~7000 GB/s ceiling the stride sweep predicted. Gated to
# power-of-two N (the rotation uses a mask) in fp32 (the bank arithmetic is 4-byte-specific);
# everything else keeps the direct load, already at SOL below the cliff.
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
    if device is not None and torch.cuda.get_device_properties(device).major < 9:
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
    # nfields is ACCEPTED (signature parity + autotuner key) but deliberately NOT acted
    # on. The nfields sweep confirmed wide-accumulator traits (var nf=3) prefer a
    # different config, BUT the fp32 and bf16 optima move in OPPOSITE directions (fp32
    # var wants larger tpr / smaller nt; bf16 var the reverse), so no single scalar
    # rule serves both -- a measured `eff = N*nfields` fit REGRESSED bf16 var to ~0.92x
    # while helping fp32. Until a per-(dtype, nfields) table is characterized densely
    # enough to interpolate, nfields is left to the autotuner (which measures per key and
    # cannot regress). nf=1 (sum/mean/prod/norm/...) is unaffected and keeps its picks.
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
    # Occupancy override for a ONE-ROW launch (the reduce-all one-shot), or None when the
    # ladder's pick already stands (the caller then passes no override at all, keeping the
    # default path -- including its ragged-N bucket gate -- untouched).
    #
    # The ladder's small tpr exists so that nt//tpr ROWS pack into each block -- with a
    # single row there is nothing to pack, and the pick instead leaves the whole GPU running
    # a fraction of one CTA (128 threads at N=4096). Give that row the widest LADDER RUNG it
    # can FEED instead (one thread per vector load), one row per block.
    #
    # Pick from _TPR_RUNGS rather than computing a width: tpr here is both the reduce-tree
    # width and the block size, so it must be a power of two AND a warp multiple. Computing
    # it got that wrong twice -- N//vec gave 50 threads for a 100-element fp64 row (not a
    # warp multiple at all), and warp-rounding gave tpr=96, three warps, which silently
    # returned a WRONG variance for a 400-element fp32 reduce-all. Rungs are valid by
    # construction, and a row too narrow for one warp keeps the ladder's pick.
    #
    # MEASURED (M=1, bf16, vs ATen): var_mean 0.53-0.93x -> 1.47-1.62x for N=1024..4096;
    # max.dim, aminmax and sum match or improve at every N from 64 to 16384. A flat
    # tpr=nt=256 also fixes mid-N but REGRESSES N<=256 (0.76x on var_mean), which is why
    # this feeds the row rather than maxing it out.
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

    key = ("rowtile", trait_key, x.dtype, tuple(out_dtypes[:nouts])) + op.cache_sig
    build = lambda: _compile(op, *_wrap())  # noqa: E731
    fn = cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")
    fn(*_wrap())
    return tuple(outs)
