# ROW reductions: launch policy for tile.TileReduce on the row axis (reduce the contiguous
# last dim of a (M, N) input). The kernel body lives in tile.py -- this module owns the
# measured launch shapes, the narrow-row gates, and the plan cache.
#
# Covers 1 or 2 outputs and either a final projection or raw stage-1 partials (which is what
# makes it reusable as the cross-CTA driver's stage 1), with a ROLLED chunk loop so ONE
# compiled kernel serves every N in a vec class -- tpr and nt are compile-time while N itself
# arrives as a runtime arg.
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
):
    """Tile-based row reduction: reduce the contiguous last dim of a 2D `x` -> (M,).

    Returns a tuple of `nouts` outputs.
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
    )

    # final -> nouts projected results; stage 1 -> one RAW partial buffer per trait field
    ndst = nouts if final else trait.nfields
    outs = [torch.empty(M, device=x.device, dtype=dt) for dt in out_dtypes[:ndst]]
    nchunks = Int32(N // op.vec)
    nwaves = Int32(math.ceil((N // op.vec) / tpr))
    # Declared alignment is what lets the load stage emit the wide instruction; tile owns the
    # derivation so it cannot be forgotten here (it was, and cost 3x). The fold takes N at
    # RUNTIME, so this wraps with BOTH extents dynamic (N carrying divisibility=vec so the
    # wide load survives) and one compiled kernel serves the whole vec class.
    align = tile.align_bytes(N, x.element_size())

    def _wrap():
        # q/npar belong to the COL axis: None, not a dummy value -- an unused Int32 param
        # costs real time (see tile.TileReduce.kernel).
        return (
            [_L.cute_tensor_dynMN(x, op.vec, align=align, read_only=True)],
            [_L.cute_tensor_dynM(o, ndim=1) for o in outs],
            nchunks,
            nwaves,
            Int32(N),
            None,
            None,
            _stream(),
        )

    key = ("rowtile", trait_key, x.dtype, tuple(out_dtypes[:ndst])) + op.cache_sig
    build = lambda: _compile(op, *_wrap())  # noqa: E731
    fn = cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")
    fn(*_wrap())
    return tuple(outs)
