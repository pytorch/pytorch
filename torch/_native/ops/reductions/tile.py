# The shared reduction DATAPATH: where a row's load width, alignment and thread mapping are
# derived, plus the folds that walk them.
#
# Why this module exists. The load stage is where the bugs were, twice, in two hand-rolled
# copies: one kernel lost 3.7x to per-element (un-widened) reads, and another then lost 3x to
# a missing `assumed_align` -- both invisible in the source. One load stage means one place to
# get width and alignment right, and one place to record WHY (see vec_size / align_bytes).
#
# A tile is described by TileMap: `vec` elements per load, `tpr` threads per row. `tpr == 1`
# is the degenerate multirow shape -- one thread owns a whole row and there is no lane merge
# at all -- and any tpr > 1 shape finishes by folding across its lanes (merge_lanes).
#
# The folds here are ROLLED: the trip count is a RUNTIME value, so ONE compiled kernel serves
# every row length in a vec class. That is a requirement, not a preference -- a static
# per-thread loop makes compile time scale with the shape (see MAX_UNROLL) and the kernel
# count scale with the number of distinct shapes seen.

import math

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64


WARP = 32

# HARD bound on the per-thread unroll, enforced in TileMap. A tile of `vec * loads` elements
# per thread folded with a STATIC trip count emits that whole loop at trace time, so COMPILE
# TIME SCALES WITH it. Measured (fp32 sum, cold compile, one N per point):
#
#   unrolled ops   12    80   320   640  1280  2560
#   compile (s)  0.14  0.17  0.35  0.60  1.17  4.54
#
# so roughly 1 ms per unrolled element op, going superlinear past ~1300. A multi-field trait
# multiplies the IR per element (Welford is 3 fields), so the effective cost is worse. Left
# unbounded, a caller asking for N=4096 at tpr=1 emits 4096 element ops and compilation does
# not finish in any reasonable time -- that is not a hypothetical, it wedged a sweep here.
#
# This is a SAFETY bound, deliberately looser than any perf gate. The point is that no caller
# -- including a benchmark harness that bypasses the perf gates -- can silently blow up
# compile time.
MAX_UNROLL = 512


def vec_size(N: int, itemsize: int) -> int:
    """Elements per load instruction.

    gcd rather than `16 // itemsize`: it makes vec divide N, which buys three things at
    once -- no ragged tail inside a chunk, every chunk base a multiple of vec, and a row
    stride (N*itemsize) that is a multiple of vec*itemsize, so every ROW start carries the
    same alignment as the base pointer.
    """
    return math.gcd(N, max(1, 16 // itemsize))


def align_bytes(N: int, itemsize: int) -> int:
    """Alignment to DECLARE on the input wrap so the DSL emits the wide load.

    Not optional: `from_dlpack` otherwise assumes only the element's natural width and
    silently emits narrow loads (measured 3x on the multirow shape). Safe by the `vec_size`
    argument above, given a tensor whose base pointer is an allocation base.
    """
    return vec_size(N, itemsize) * itemsize


class TileMap:
    """How one row is spread over threads and loads.

    tpr == 1 -> one thread owns a whole row, and there is no lane merge.
    """

    def __init__(self, N: int, itemsize: int, tpr: int, loads: int):
        if tpr != 1 and tpr % WARP != 0:
            raise ValueError(f"tpr must be 1 or a multiple of {WARP}, got {tpr}")
        unroll = vec_size(N, itemsize) * loads
        if unroll > MAX_UNROLL:
            raise ValueError(
                f"per-thread unroll {unroll} (vec*loads) exceeds MAX_UNROLL={MAX_UNROLL}; "
                f"compile time scales with it (~1 ms/op, superlinear past ~1300). "
                f"Got N={N} tpr={tpr} loads={loads}."
            )
        self.N = N
        self.vec = vec_size(N, itemsize)
        self.tpr = tpr
        self.loads = loads
        # A wide load needs vec to divide N, which is what makes every row start (row stride
        # N*itemsize) and every chunk base carry the base pointer's alignment. When it does
        # not, the load falls back to per-element reads.
        self.wide_ok = N % self.vec == 0

    @property
    def sig(self):
        return (self.N, self.vec, self.tpr, self.loads, self.wide_ok)

    def align_bytes(self, itemsize: int) -> int:
        """Alignment to declare for THIS tile (element width when the wide load is off)."""
        return self.vec * itemsize if self.wide_ok else itemsize


@cute.jit
def merge_lanes(trait, acc, tm: cutlass.Constexpr, asc: cutlass.Constexpr = False):
    """Reduce across the `tpr` lanes covering one row. A no-op at tpr == 1.

    `asc` selects the ASCENDING butterfly over the descending one. The folds below hand
    columns out in that direction, and an index trait's ties depend on which it is.
    """
    if const_expr(tm.tpr == 1):
        return acc
    from .._cutedsl.traits import warp_reduce

    return warp_reduce(trait, acc, tm.tpr, ascending=asc)


_ROLL_UNROLL = 4


@cute.jit
def fold_row_rolled(
    trait,
    mX,
    r,
    tm: cutlass.Constexpr,
    lane,
    nchunks,
    nwaves,
    unroll: cutlass.Constexpr = _ROLL_UNROLL,
):
    """Fold row `r` across `tm.tpr` lanes with a RUNTIME chunk loop. Returns an acc tuple.

    Each wave covers tpr*vec contiguous elements; this thread takes chunk (c*tpr + lane).
    A wave that runs past the row's last chunk is handled by CLAMPING the chunk index and
    passing valid=False to the trait, not by branching -- binding a dynamic value inside a
    dynamic `if` is rejected by the DSL, and the trait already folds `valid` correctly.
    """
    reduce_fn, acc_dt = trait.reduce, trait.acc
    acc = trait.init()
    vec = const_expr(tm.vec)
    tpr = const_expr(tm.tpr)
    gv = cute.flat_divide(mX[Int64(r), None], (vec,))
    frag = cute.make_rmem_tensor(cute.make_layout(vec), mX.element_type)
    for c in cutlass.range(nwaves, unroll=unroll):
        k = c * Int32(tpr) + lane
        ok = k < nchunks
        ks = k if ok else Int32(0)  # clamp so the load is always in range
        cute.autovec_copy(gv[None, ks], frag)
        for i in cutlass.range_constexpr(vec):
            acc = reduce_fn(acc, acc_dt(frag[i]), ks * Int32(vec) + Int32(i), ok)
    return acc
