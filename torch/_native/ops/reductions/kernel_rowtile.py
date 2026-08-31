# The vectorized ROW reduction: reduce the contiguous last dim of a (M, N) input with
# `tpr` threads per row, on the shared tile datapath (tile.py).
#
# Covers 1 or 2 outputs and either a final projection or raw stage-1 partials (which is what
# makes it reusable as the cross-CTA driver's stage 1), with a ROLLED chunk loop so ONE
# compiled kernel serves every N in a vec class -- tpr is a multiple of WARP and
# rows_per_block = nt // tpr, both compile-time, while N itself arrives as a runtime arg.
#
# The launch shape (tpr, nt) is the only tuned part, and it is DATA below: a measured ladder
# plus a one-row override for the reduce-all case, both of which pick from the same set of
# legal rungs.

import math
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32

import torch

from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan
from .._cutedsl.traits import block_reduce, WARP
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


class RowTile:
    def __init__(self, trait, dtype, N, tpr, nt, nouts=1, final=True, unroll=4):
        if tpr % WARP or tpr > nt or nt % tpr:
            raise ValueError(
                f"tpr must be a multiple of {WARP} dividing nt: {tpr=} {nt=}"
            )
        self.trait = trait
        self.dtype = dtype
        self.N = N
        self.tpr = tpr
        self.nt = nt
        self.nouts = nouts
        self.final = final
        self.unroll = unroll
        self.rows_per_block = nt // tpr
        self.warps_per_row = tpr // WARP
        # loads=1: the fold is ROLLED, so the static per-thread load count is unused and the
        # MAX_UNROLL bound is trivially met -- tile is here for `vec` and the lane mapping.
        isz = dtype.width // 8
        self.tm = tile.TileMap(N, isz, tpr, 1)
        self.vec = self.tm.vec

    @property
    def cache_sig(self):
        # N is ABSENT: the rolled loop takes it at runtime, so one kernel serves the vec class.
        t = self.trait.nfields
        return (self.vec, self.tpr, self.nt, self.nouts, self.final, self.unroll, t)

    @cute.jit
    def __call__(self, mX, mOuts: list, nchunks, nwaves, project_n, stream):
        self.kernel(mX, mOuts, nchunks, nwaves, project_n).launch(
            grid=[cute.ceil_div(mX.shape[0], const_expr(self.rows_per_block)), 1, 1],
            block=[const_expr(self.nt), 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX, mOuts: list, nchunks, nwaves, project_n):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        trait = self.trait
        row = Int32(bx) * const_expr(self.rows_per_block) + Int32(
            tx // const_expr(self.tpr)
        )
        lane = Int32(tx % const_expr(self.tpr))
        # Rows past the end clamp to row 0 so every load stays in range; the store is dropped.
        alive = row < Int32(mX.shape[0])
        rs = row if alive else Int32(0)
        acc = tile.fold_row_rolled(
            trait, mX, rs, self.tm, lane, nchunks, nwaves, const_expr(self.unroll)
        )
        acc = tile.merge_lanes(trait, acc, self.tm)
        if const_expr(self.warps_per_row > 1):
            smem = cutlass.utils.SmemAllocator()
            bufs = [
                smem.allocate_tensor(
                    trait.fdtypes[f],
                    cute.make_layout(self.rows_per_block * self.warps_per_row),
                    byte_alignment=8,
                )
                for f in range(trait.nfields)
            ]
            acc = block_reduce(
                trait,
                acc,
                bufs,
                const_expr(self.warps_per_row),
                const_expr(self.rows_per_block),
            )
        # project OUTSIDE the store branch: binding a dynamic value inside a dynamic `if` is
        # rejected by the DSL (see kernel_general's final store for the same shape).
        if const_expr(self.final):
            res = trait.project(acc, trait.acc(project_n))
            if lane == 0 and alive:
                if const_expr(self.nouts == 1):
                    mOuts[0][row] = mOuts[0].element_type(res)
                else:
                    for k in cutlass.range_constexpr(self.nouts):
                        mOuts[k][row] = mOuts[k].element_type(res[k])
        else:
            if lane == 0 and alive:
                for f in cutlass.range_constexpr(trait.nfields):
                    mOuts[f][row] = trait.fdtypes[f](acc[f])


def reduce_row_tile(
    trait, trait_key, x, out_dtypes, nouts=1, tpr=None, nt=None, final=True, unroll=None
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
    op = RowTile(trait, dt, N, tpr, nt, nouts, final, unroll)

    # final -> nouts projected results; stage 1 -> one RAW partial buffer per trait field
    ndst = nouts if final else trait.nfields
    outs = [torch.empty(M, device=x.device, dtype=dt) for dt in out_dtypes[:ndst]]
    align = tile.align_bytes(N, x.element_size())
    nchunks = Int32(N // op.vec)
    nwaves = Int32(math.ceil((N // op.vec) / tpr))

    def _wrap():
        return (
            _L.cute_tensor_dynMN(x, op.vec, align=align, read_only=True),
            [_L.cute_tensor_dynM(o, ndim=1) for o in outs],
            nchunks,
            nwaves,
            Int32(N),
            _stream(),
        )

    key = ("rowtile", trait_key, x.dtype, tuple(out_dtypes[:ndst])) + op.cache_sig
    build = lambda: _compile(op, *_wrap())  # noqa: E731
    fn = cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")
    fn(*_wrap())
    return tuple(outs)
