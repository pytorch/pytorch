# The shared reduction DATAPATH: where a row's load width, alignment and thread mapping are
# derived, plus the folds that walk them. It exists because the load stage is where the bugs
# were, twice, in two hand-rolled copies -- one lost 3.7x to un-widened reads and another 3x to
# a missing assumed_align, both invisible in the source.
#
# A tile is `vec` elements per load and `tpr` threads per row; tpr == 1 is the degenerate shape
# with no lane merge at all. The folds are ROLLED, which is a requirement rather than a
# preference: a static per-thread loop makes compile time scale with the shape and the kernel
# count with the number of distinct shapes seen.

import math

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64


WARP = 32

# SAFETY bound on the per-thread unroll, enforced in TileMap: a static trip count is emitted
# at trace time, so compile time scales with it and turns superlinear past ~1300 ops.
MAX_UNROLL = 512


def vec_size(N: int, itemsize: int) -> int:
    """Elements per load instruction. gcd, not `16 // itemsize`, so vec DIVIDES N: no ragged
    tail in a chunk, and every chunk base carries the base pointer's alignment.
    """
    return math.gcd(N, max(1, 16 // itemsize))


def align_bytes(N: int, itemsize: int) -> int:
    """Alignment to DECLARE on the input wrap. Not optional: from_dlpack otherwise assumes the
    element width and silently emits narrow loads, measured 3x on the multirow shape.
    """
    return vec_size(N, itemsize) * itemsize


class TileMap:
    """How one row is spread over threads and loads. tpr == 1 means one thread owns a whole row,
    with no lane merge.
    """

    def __init__(self, N: int, itemsize: int, tpr: int, loads: int):
        # The warp COUNT must be a power of two, not just a multiple of 32: the cross-warp
        # butterfly spans tpr // WARP groups only then, and silently drops a partial otherwise.
        nw = tpr // WARP
        if tpr != 1 and (tpr % WARP or nw & (nw - 1)):
            raise ValueError(
                f"tpr must be 1 or a power-of-two multiple of {WARP}, got {tpr}"
            )
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
        # A wide load needs vec to divide N, which is what makes every row start and chunk base carry
        # the base pointer's alignment. Otherwise the load falls back to per-element reads.
        self.wide_ok = N % self.vec == 0

    @property
    def sig(self):
        return (self.N, self.vec, self.tpr, self.loads, self.wide_ok)

    def align_bytes(self, itemsize: int) -> int:
        """Alignment to declare for THIS tile (element width when the wide load is off)."""
        return self.vec * itemsize if self.wide_ok else itemsize


@cute.jit
def merge_lanes(trait, acc, tm: cutlass.Constexpr, asc: cutlass.Constexpr = False):
    """Reduce across the `tpr` lanes covering one row; a no-op at tpr == 1.

    `asc` selects the ASCENDING butterfly, which the folds' column order depends on and an
    index trait's ties depend on.
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

    A wave past the row's last chunk CLAMPS its index and passes valid=False rather than
    branching, which the DSL rejects for a dynamic bind.
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
