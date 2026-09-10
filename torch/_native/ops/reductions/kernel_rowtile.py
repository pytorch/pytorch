# Vectorized reduction of contiguous (M, N) rows, with 1-2 projected outputs or raw
# per-field partials. Its rolled loop shares one kernel per vector class. Launch shape
# comes from measured legal-rung and reduce-all one-row configurations below.

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


# First matching B200 threads-per-row anchor wins; small rows pack without cross-warp merge.
_TPR_LADDER = ((64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128))
_TPR_MAX = 256
# Legal power-of-two reduction/block widths, widest last.
_TPR_RUNGS = tuple(t for _, t in _TPR_LADDER) + (_TPR_MAX,)
# Small rows use fewer threads/block.
_NT_SMALL, _NT_LARGE, _NT_GATE_N = 128, 256, 16 * 1024
# Rows >=16 KB need 256 threads; the element ladder underthreads them by 1.1-1.4x.
_WIDE_ROW_BYTES = 16 * 1024


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


class RowTile:
    def __init__(self, trait, dtype, N, tpr, nt, nouts=1, final=True, unroll=4):
        # block_reduce drops partials unless the row's warp count is a power of two.
        nw = tpr // WARP
        if tpr % WARP or tpr > nt or nt % tpr or nw & (nw - 1):
            raise ValueError(
                f"tpr must be a power-of-two multiple of {WARP} dividing nt: {tpr=} {nt=}"
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
        # Rolled folding needs TileMap only for vec and lane mapping.
        isz = dtype.width // 8
        self.tm = tile.TileMap(N, isz, tpr, 1)
        self.vec = self.tm.vec

    @property
    def cache_sig(self):
        # Runtime N lets one kernel serve the vector class.
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
        # Project before the store branch; the DSL rejects dynamic values bound inside it.
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
    trait, trait_key, x, out_dtypes, nouts=1, tpr=None, nt=None, final=True, unroll=None
):
    """Reduce 2-D `x` rows, returning outputs or raw field partials when `final=False`."""
    if x.dim() != 2 or not x.is_cuda or x.stride(-1) != 1:
        raise AssertionError(f"want 2D contiguous-last-dim CUDA, got {tuple(x.shape)}")
    M, N = x.shape
    cfg = row_config(N, x.element_size() * 8)
    # Scalar rows use 16 to hide narrow-load latency; vectorized rows use 4, at or near
    # the measured optimum across 4/8/16/32.
    if unroll is None:
        unroll = 16 if tile.vec_size(N, x.element_size()) == 1 else 4
    tpr = max(WARP, cfg.tpr) if tpr is None else tpr
    nt = max(tpr, cfg.nt) if nt is None else nt
    nt -= nt % tpr  # rows_per_block must be whole
    dt = torch2cute[x.dtype]
    op = RowTile(trait, dt, N, tpr, nt, nouts, final, unroll)

    # Final projects nouts; stage 1 stores one raw buffer per field.
    ndst = nouts if final else trait.nfields
    outs = [torch.empty(M, device=x.device, dtype=dt) for dt in out_dtypes[:ndst]]
    # Narrow N-derived alignment to what the base pointer meets.
    align = _declared_align(x, tile.align_bytes(N, x.element_size()))
    nchunks = Int32(N // op.vec)
    nwaves = Int32(math.ceil((N // op.vec) / tpr))

    def _fake():
        # Dynamic row-major extents share a vector-class kernel. Baked alignment keeps loads
        # wide; None omits costly unused-axis arguments.
        return (
            _L.fake_compact(dt, (_L.sym(), _L.sym(op.vec)), order=(1, 0), align=align),
            [_L.fake_compact(torch2cute[o.dtype], (_L.sym(),)) for o in outs],
            nchunks,
            nwaves,
            Int32(N),
            _stream(),
        )

    # Pointer-dependent alignment is baked, so include it in the key.
    dts = tuple(out_dtypes[:ndst])
    key = ("rowtile", trait_key, x.dtype, dts, align) + op.cache_sig
    build = lambda: _compile(op, *_fake())  # noqa: E731
    fn = cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")
    # read_only avoids COW materialization; None omits unused axes.
    fn(_L.read_only(x), list(outs), nchunks, nwaves, Int32(N), _stream())
    return tuple(outs)
