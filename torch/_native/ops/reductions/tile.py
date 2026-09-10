# Shared row load width, alignment, thread map, and folds. Reuse avoids prior 3.7x
# unwidened-load and 3x undeclared-alignment regressions. A tile has vec elements/load
# and tpr threads/row; rolled folds prevent compile time and kernel count scaling by shape.

import math

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64


WARP = 32

# Static unroll compiles superlinearly past ~1300 operations.
MAX_UNROLL = 512


def vec_size(N: int, itemsize: int) -> int:
    """Elements/load; gcd makes vec divide N and preserves alignment at every chunk."""
    return math.gcd(N, max(1, 16 // itemsize))


def align_bytes(N: int, itemsize: int) -> int:
    """Input alignment declaration; omitting it emits narrow loads and costs 3x."""
    return vec_size(N, itemsize) * itemsize


class TileMap:
    """Map row elements to threads and loads; tpr=1 needs no lane merge."""

    def __init__(self, N: int, itemsize: int, tpr: int, loads: int):
        # Cross-warp butterflies require a power-of-two warp count.
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
        # Wide loads require vec to divide N; otherwise use per-element reads.
        self.wide_ok = N % self.vec == 0

    @property
    def sig(self):
        return (self.N, self.vec, self.tpr, self.loads, self.wide_ok)

    def align_bytes(self, itemsize: int) -> int:
        """Alignment to declare for THIS tile (element width when the wide load is off)."""
        return self.vec * itemsize if self.wide_ok else itemsize


@cute.jit
def merge_lanes(trait, acc, tm: cutlass.Constexpr, asc: cutlass.Constexpr = False):
    """Merge row lanes; `asc` controls numeric association and index ties. No-op at tpr=1."""
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
    """Fold row `r` with a runtime loop, clamping and masking tail waves to avoid DSL branches."""
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
