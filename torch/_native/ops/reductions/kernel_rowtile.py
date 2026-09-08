# The vectorized ROW reduction: the contiguous last dim of an (M, N) input with `tpr` threads
# per row, on the shared tile datapath. Covers 1 or 2 outputs and either a final projection or
# raw stage-1 partials -- which is what makes it reusable as the cross-CTA driver's stage 1 --
# with a ROLLED chunk loop, so one compiled kernel serves every N in a vec class.
#
# The launch shape is the only tuned part, and it is DATA below: a measured ladder plus a
# one-row override for reduce-all, both picking from the same legal rungs.

import math
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32

import torch

from ...cutedsl.dtypes import torch2cute
from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan
from .._cutedsl.traits import block_reduce, WARP
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


class _RowConfig(NamedTuple):
    # Row-reduce knob set: knobs left None are filled from row_config, and explicit values
    # override per field. This is the row kernel's OWN config -- the other axes differ in shape.
    tpr: int  # threads per row
    nt: int  # threads per block


def row_config(N: int, dtype_width: int, nfields: int = 1, hw=None) -> "_RowConfig":
    # Occupancy config from (N, dtype). The ladder's N-limits are proxies for how wide a row gets
    # before it needs more threads, which tracks smem capacity, so scaling them by hw.smem_scale
    # keeps the rule right on other GPUs. nfields is accepted for signature parity but NOT acted
    # on: the fp32 and bf16 optima move in OPPOSITE directions, so no scalar rule serves both.
    smem = 1.0 if hw is None else hw.smem_scale
    # Wide-row rung first, and byte-based: a >=16KB row saturates 256 threads whatever the dtype.
    # This overrides the element ladder, which under-threads mid-N wide rows by ~1.3x.
    if N * (dtype_width // 8) >= _WIDE_ROW_BYTES * smem:
        return _RowConfig(tpr=_TPR_MAX, nt=_NT_LARGE)
    tpr = next((t for limit, t in _TPR_LADDER if N <= limit * smem), _TPR_MAX)
    nt = _NT_SMALL if N <= _NT_GATE_N * smem else _NT_LARGE
    return _RowConfig(tpr=tpr, nt=nt)


def single_row_config(N: int, dtype_width: int, nfields: int = 1, hw=None):
    # Occupancy override for a ONE-ROW launch, or None to leave the ladder's pick standing. The
    # ladder's small tpr exists so rows pack per block; with one row the GPU runs a fraction of
    # one CTA, so give that row the widest rung it can feed. From _TPR_RUNGS rather than a
    # computed width, since tpr is both tree width and block size -- a computed one returned a
    # wrong variance. Measured 0.53-0.93x -> 1.47-1.62x of ATen on var_mean.
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

    Returns a tuple of `nouts` outputs. tpr=1 is the NARROW-row shape, TMA-staged where that
    wins. `order` selects the fold order; see itree_plan for the reproducible one.
    """
    if x.dim() != 2 or not x.is_cuda or x.stride(-1) != 1:
        raise AssertionError(f"want 2D contiguous-last-dim CUDA, got {tuple(x.shape)}")
    M, N = x.shape
    cfg = row_config(N, x.element_size() * 8, trait.nfields)
    # Unroll depth of the rolled wave loop. A SCALAR row (an odd or prime N) has no wide load to
    # hide latency behind and wants more loads in flight; a vectorized row pays for the depth.
    # Measured across unroll 4/8/16/32, 4 is at or within noise of the best at every shape.
    if unroll is None:
        unroll = 16 if tile.vec_size(N, x.element_size()) == 1 else 4
    tpr = max(WARP, cfg.tpr) if tpr is None else tpr
    nt = max(tpr, cfg.nt) if nt is None else nt
    nt -= nt % tpr  # rows_per_block must be whole
    dt = torch2cute[x.dtype]
    op = RowTile(trait, dt, N, tpr, nt, nouts, final, unroll)

    # final -> nouts projected results; stage 1 -> one RAW partial buffer per trait field
    ndst = nouts if final else trait.nfields
    outs = [torch.empty(M, device=x.device, dtype=dt) for dt in out_dtypes[:ndst]]
    align = tile.align_bytes(N, x.element_size())
    nchunks = Int32(N // op.vec)
    nwaves = Int32(math.ceil((N // op.vec) / tpr))

    def _fake():
        # Compile-time descriptors: 2D row-major with both extents dynamic, the inner one divisible by
        # vec so one kernel serves the vec class. `align` is what keeps the load wide, narrowed to what
        # the base pointer meets. The col axis's args are None, not dummies -- an unused Int32 param
        # costs real time.
        return (
            _L.fake_compact(dt, (_L.sym(), _L.sym(op.vec)), order=(1, 0), align=align),
            [_L.fake_compact(torch2cute[o.dtype], (_L.sym(),)) for o in outs],
            nchunks,
            nwaves,
            Int32(N),
            _stream(),
        )

    key = ("rowtile", trait_key, x.dtype, tuple(out_dtypes[:ndst])) + op.cache_sig
    build = lambda: _compile(op, *_fake())  # noqa: E731
    fn = cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")
    # The real operands: read_only on the INPUT, or a COW input materializes on export. The other
    # axes' args are None rather than dummies -- an unused Int32 param costs real time.
    fn(_L.read_only(x), list(outs), nchunks, nwaves, Int32(N), _stream())
    return tuple(outs)
