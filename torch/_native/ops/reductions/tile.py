# The shared reduction KERNEL and the datapath it is built from: where a tile's load width,
# alignment and thread mapping are derived, the folds that walk them, and the ONE @cute.kernel
# (TileReduce, at the bottom) that every fast reduction path launches -- row or column,
# one-shot or split stage, direct or TMA-staged. The kernel_* modules above it are drivers:
# they pick the launch shape and own the plan cache, but the body lives here.
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
# Most folds here are ROLLED: the trip count is a RUNTIME value, so ONE compiled kernel
# serves every row length in a vec class. That is the default for a good reason -- a static
# per-thread loop makes compile time scale with the shape (see MAX_UNROLL) and the kernel
# count scale with the number of distinct shapes seen.
#
# The exception is an ORDER that fixes its add DAG at compile time (the inner-tree order): it
# binds every value to a slot in a tree, so it cannot ride a runtime trip count. That form
# needs the STATIC fragment -- `load` fills a per-thread tile once and the fold walks it with
# `range_constexpr` -- and pays the compile/kernel-count cost in exchange for a reproducible
# bit pattern. `TileMap.strides` is where the two chunk-to-warp assignments differ, and it is
# the only place they differ: the emitted loads are identical either way. The one thing that
# order still takes at RUNTIME is the number of per-row partials its widest shape leaves behind
# -- that count grows with N, and a linear chain folds in the same order however it is rolled.

import math
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64, pipeline
from cutlass.cute.nvgpu import cpasync


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


def _magic(d):
    # Magic-number reciprocal for exact n // d over 0 <= n < 2^31 as
    # (n * m) >> sh -- one 64-bit multiply + shift instead of a runtime 64-bit
    # divide (which the ~%25-slower runtime-geometry decode would otherwise emit
    # per element per pair). Same Granlund-Montgomery family as aten's IntDivider
    # (aten/src/ATen/cuda/detail/IntegerDivider.cuh, hackersdelight.org/magic.htm);
    # that one uses the add-indicator form for the full unsigned 2^32 domain, but
    # K0's linear indices are Int32-positive (< 2^31), so the simpler round-up form
    # is exact and one instruction cheaper. Proof sketch: m = floor(2^(31+l)/d)+1
    # with l = ceil(log2 d), so m*d = 2^(31+l) + e with 0 < e <= d, and for n < 2^31
    # the error term n*e/(d*2^(31+l)) < 2^-l * 1 < 1/d ... floor((n*m) >> (31+l))
    # = n//d exactly. m < 2^32 (d > 2^(l-1)) so n*m < 2^63: no Int64 overflow.
    l = (d - 1).bit_length()
    return (1 << (31 + l)) // d + 1, 31 + l


def _decode_offset(linear, vals, npairs):
    # Mixed-radix decode of a linear index into a flat element offset. vals is a
    # RUNTIME Int64 list of QUADS, fastest-varying dim first:
    #     [m0, sh0, ext0, strd0,  m1, sh1, ext1, strd1, ...]
    # where (m, sh) is _magic(ext). npairs is the compile-time pair COUNT (only the
    # loop STRUCTURE is baked -- the values are launch args, so one compiled kernel
    # serves every geometry with the same pair count). Divisions run as magic
    # multiply+shift; the LAST pair needs neither div nor mod (a linear index in
    # range has rem < ext_last; out-of-range lanes decode garbage that the callers'
    # `valid` predication never reads). For a single pair this is linear*stride.
    #
    # INT64: the flat offset can exceed int32 (numel >= 2^31, e.g. a (300000, 8192)
    # reduction). Cast the linear index to Int64 up front so every rem*stride product
    # and accumulation is 64-bit; an int32 product silently wraps negative and reads
    # out of bounds. The returned offset indexes a flat gmem tensor, which expects a
    # 64-bit offset.
    rem = cutlass.Int64(linear)
    # npairs is at least 1 here: an empty KEPT list is legal (a full reduction) but the
    # caller drops the decode entirely for it, and a plan with no REDUCED runs is refused by
    # ReduceBlock. Zero pairs would index vals[-1] below.
    if npairs == 1:
        return rem * vals[3]
    off = cutlass.Int64(0)
    for j in range(npairs - 1):
        q = (rem * vals[4 * j]) >> vals[4 * j + 1]
        off = off + (rem - q * vals[4 * j + 2]) * vals[4 * j + 3]
        rem = q
    return off + rem * vals[4 * (npairs - 1) + 3]


def _ilog2(n: int) -> int:
    return max(1, n.bit_length() - 1)


def _off(base, i: int):
    # Keep a static base static (see TileMap.col_base): int + int stays foldable.
    return base + i if isinstance(base, int) else base + Int32(i)


class TileMap:
    """How one row is spread over threads and loads.

    tpr == 1 -> one thread owns a whole row, and there is no lane merge.
    """

    def __init__(
        self,
        N: int,
        itemsize: int,
        tpr: int,
        loads: int,
        warp_major: bool = False,
        vec: int | None = None,
        exact: bool | None = None,
    ):
        if tpr != 1 and tpr % WARP != 0:
            raise ValueError(f"tpr must be 1 or a multiple of {WARP}, got {tpr}")
        unroll = (vec_size(N, itemsize) if vec is None else vec) * loads
        if unroll > MAX_UNROLL:
            raise ValueError(
                f"per-thread unroll {unroll} (vec*loads) exceeds MAX_UNROLL={MAX_UNROLL}; "
                f"compile time scales with it (~1 ms/op, superlinear past ~1300). "
                f"Got N={N} tpr={tpr} loads={loads}."
            )
        self.N = N
        # `vec` is normally derived, but an order can DEFINE itself in terms of 16 // itemsize
        # regardless of N (the inner-tree order does, and identity-pads a ragged row), so that
        # caller passes its own -- the derived gcd form would change the add DAG and the bits.
        self.vec = vec_size(N, itemsize) if vec is None else vec
        self.tpr = tpr
        self.loads = loads
        self.warp_major = warp_major
        self.nw = 1 if tpr == 1 else tpr // WARP
        # Exact when the tile covers the row with nothing left over: every load is then
        # unconditionally in range and no predication is emitted at all. A BATCHED tile covers
        # only its batch, so that caller passes exact=False.
        self.exact = (self.vec * self.loads * self.tpr == N) if exact is None else exact
        # A wide load needs vec to divide N, which is what makes every row start (row stride
        # N*itemsize) and every chunk base carry the base pointer's alignment. When it does
        # not, the load falls back to per-element reads.
        self.wide_ok = N % self.vec == 0

    @property
    def sig(self):
        return (
            self.N,
            self.vec,
            self.tpr,
            self.loads,
            self.warp_major,
            self.exact,
            self.wide_ok,
        )

    def align_bytes(self, itemsize: int) -> int:
        """Alignment to declare for THIS tile (element width when the wide load is off)."""
        return self.vec * itemsize if self.wide_ok else itemsize

    def strides(self):
        """(lane, w, l) column strides -- the ORDER lives here and nowhere else.

        The regular and inner-tree orders read the SAME elements into the SAME registers and
        differ only in which chunk goes to which warp, i.e. in the `l` and `w` strides being
        swapped. Both keep stride-1 innermost and both are compact, so the emitted instruction
        stream is byte-identical either way.
        """
        if self.tpr == 1:
            return (0, 0, self.vec)
        wle = WARP * self.vec  # columns one warp covers in one load
        if self.warp_major:
            return (self.vec, wle * self.loads, wle)
        return (self.vec, wle, wle * self.nw)

    def col_base(self, lane, w, l: int, warp_stride=None):
        """Column of element 0 of this thread's load `l`.

        Returns a PYTHON INT when the whole offset is compile-time (tpr == 1), and only builds
        a dynamic value when lane/w actually participate. That distinction is load-bearing,
        not cosmetic: a static offset folds into the address so the compiler can prove the
        16-byte alignment and emit the wide load, while wrapping the same number in Int32
        hides it and silently costs 3x.
        """
        s_lane, s_w, s_l = self.strides()
        if warp_stride is not None:
            # Caller supplies the per-warp stride; 0 means the warp offset is already folded
            # into base_col.
            s_w = warp_stride
        if self.tpr == 1:
            return l * s_l
        return lane * Int32(s_lane) + w * s_w + Int32(const_expr(l * s_l))


@cute.jit
def fold_decoded(
    trait,
    mX,
    obase,
    rvals,
    npairs: cutlass.Constexpr,
    rb,
    nt: cutlass.Constexpr,
    tidx,
    in_base,
    chunk_base,
    gidx: cutlass.Constexpr = "r",
):
    """Grid-stride fold of ONE output's reduced run, addressed by mixed-radix DECODE.

    The arm for an arbitrary layout: `rvals` carries the reduced (extent, stride) quads, so a
    transposed, sliced, permuted, expanded or overlapping-window input is just a different
    decode rather than a different kernel. All `nt` threads of the block cooperate on this one
    output and their partial accumulators are merged by the caller.

    `rb` is the pre-clamped step bound (the caller applies the reduce-all tail or the ragged
    chunk clamp), so every FULL wave is in range and its trip count can be a dynamic
    `cutlass.range` -- a per-element `valid` would trip the IR flattener. One predicated
    remainder pass follows, and compile depth stays O(1) in the extent.

    `gidx` picks what an index trait is told the position is: "r" the linear step index,
    "flat" the global flat input offset (reduce-all), "chunk" chunk_base + r (a split run).
    """
    reduce_fn, acc_dt = trait.reduce, trait.acc
    acc = trait.init()
    n_full = rb // nt
    base_r = tidx
    for _ in cutlass.range(n_full):
        # Inline the offset: an intermediate name would be treated as loop-carried. acc and
        # base_r are the only carried values, both initialized above.
        if const_expr(gidx == "flat"):
            acc = reduce_fn(
                acc,
                acc_dt(mX[obase + _decode_offset(base_r, rvals, npairs)]),
                Int32(obase + _decode_offset(base_r, rvals, npairs)),
                True,
            )
        elif const_expr(gidx == "chunk"):
            acc = reduce_fn(
                acc,
                acc_dt(mX[obase + _decode_offset(base_r, rvals, npairs)]),
                chunk_base + base_r,
                True,
            )
        else:
            acc = reduce_fn(
                acc,
                acc_dt(mX[obase + _decode_offset(base_r, rvals, npairs)]),
                base_r,
                True,
            )
        base_r = base_r + nt
    # Invalid lanes read in_base (always in range): obase itself can be past the end for an
    # overhanging chunk, whose rb clamps to 0.
    valid = base_r < rb
    off = obase + _decode_offset(base_r, rvals, npairs)
    off_s = off if valid else in_base
    val = acc_dt(mX[off_s])
    if const_expr(gidx == "flat"):
        return reduce_fn(acc, val, Int32(off_s), valid)
    if const_expr(gidx == "chunk"):
        return reduce_fn(acc, val, chunk_base + base_r, valid)
    return reduce_fn(acc, val, base_r, valid)


@cute.jit
def fold_partials_run(trait, mIns, obase, rb, nt: cutlass.Constexpr, tidx, in_base):
    """COMBINE one output's pre-reduced accumulator tuples: a contiguous run of `rb` per
    field from `obase`, grid-strided by `nt`.

    The stage-2 reader for every split whose partials are laid out per output (reduce-all,
    the ragged row split, the general split, xcta). Offsetting by obase is what keeps each
    output on its OWN partials -- without it every output reads output 0's.

    Dynamic full-wave loop + constexpr remainder, as in fold_decoded: `rb` can be ~1e5 for a
    huge-N split, and a static unroll over it scaled compile time with the partial count
    (98125 partials was a ~384-deep unroll, ~3s to compile).
    """
    combine_fn = trait.combine
    fdtypes = trait.fdtypes
    nf = const_expr(trait.nfields)
    acc = trait.init()
    n_full = rb // nt
    r = tidx
    for _ in cutlass.range(n_full):
        rr = obase + cutlass.Int64(r)
        # A bare-range comprehension unrolls at trace time, leaving no `trait` attribute
        # access inside the dynamic loop (which would trip the IR flattener).
        acc = combine_fn(acc, tuple(fdtypes[f](mIns[f][rr]) for f in range(nf)))
        r = r + nt
    valid = r < rb
    rr = (obase + cutlass.Int64(r)) if valid else in_base
    part = tuple(fdtypes[f](mIns[f][rr]) for f in range(nf))
    merged = combine_fn(acc, part)
    return tuple((merged[f] if valid else acc[f]) for f in range(nf))


@cute.jit
def merge_lanes(trait, acc, tpr: cutlass.Constexpr, asc: cutlass.Constexpr = False):
    """Reduce across the `tpr` lanes covering one output. A no-op at tpr == 1.

    `asc` selects the ASCENDING butterfly over the descending one. The folds below hand
    columns out in that direction, and an index trait's ties depend on which it is.
    """
    if const_expr(tpr == 1):
        return acc
    from .._cutedsl.traits import warp_reduce

    return warp_reduce(trait, acc, tpr, ascending=asc)


def leaf_op(trait):
    # The combiner, taken FROM THE TRAIT so ANY trait shares these orders -- accumulators are
    # nfields-tuples throughout, which is what lets an index or Welford trait use a tree fold
    # (its `reduce` fuses transform-and-combine and cannot be paired up; see traits.py).
    return trait.combine


def identity(trait):
    return trait.init()


def _linear_reduce(vals, op):
    acc = vals[0]
    for i in range(1, len(vals)):
        acc = op(acc, vals[i])
    return acc


def _inner_tree_reduce(vals, op):
    # Stride-doubling pairwise tree: stride 1, 2, 4, ... folding v[i] += v[i + stride].
    v = list(vals)
    n = len(v)
    stride = 1
    while stride < n:
        i = 0
        while i + stride < n:
            v[i] = op(v[i], v[i + stride])
            i += stride * 2
        stride *= 2
    return v[0]


def _reduce_vec(vals, vec, op):
    return _inner_tree_reduce(vals, op) if vec >= 4 else _linear_reduce(vals, op)


def _streaming_push(tree, val, load: int, max_depth: int, op) -> None:
    # ATen's streaming_inner_tree_step. The merge count is the number of trailing zero bits of
    # (load + 1) -- __ffs(load + 1) - 1 -- capped at max_depth. The existing accumulator stays
    # on the LEFT: carry = op(tree.pop(), carry). Both details are load-bearing for bitwise
    # equality.
    trailing_zeros = ((load + 1) & -(load + 1)).bit_length() - 1
    carry = val
    for _ in range(min(trailing_zeros, max_depth)):
        carry = op(tree.pop(), carry)
    tree.append(carry)


@cute.jit
def fold_groups(
    trait,
    frag,
    cols,
    vec: cutlass.Constexpr,
    hi,
    max_depth: cutlass.Constexpr,
    merge_tpr: cutlass.Constexpr,
    merge_per_group: cutlass.Constexpr = True,
    exact: cutlass.Constexpr = False,
    vec_linear: cutlass.Constexpr = False,
):
    """THE inner-tree fold. One body for every shape of the order; the caller supplies the layout.

    `frag` is a (vec, ngroups) register tile and `cols[i]` is the true column of group i's element
    0 -- that pair is the whole interface, so a strided per-chunk read and a contiguous run read
    back from smem both land here unchanged. Returns an nfields acc tuple.

    Two parameters carry the only structural differences between the callers:
      * `merge_per_group` -- the lane butterfly runs INSIDE the group loop (one per 32-lane load,
        which is what ATen's per-chunk nesting does) or ONCE at the end (which is what a lane
        owning a contiguous run allows). Both are the balanced tree over the same columns, so the
        bits are the same either way; the choice is only where the shuffles happen.
      * `exact` -- when the tile provably lies inside the row the mask is a no-op, and emitting it
        would cost a compare and a select PER ELEMENT on the fold's critical path.

    Works for ANY trait, because every step goes through `leaf` and `combine` on nfields-tuples: an
    index trait gets its position from the column the tree already knows, and Welford's
    single-element accumulators merge pairwise (its parallel formula, not its online one).

    A slot past the row's end folds the IDENTITY rather than being skipped -- that padding is part
    of the DAG, not an implementation detail.
    """
    op, ident = leaf_op(trait), identity(trait)
    nf = const_expr(trait.nfields)
    tree: list = []
    for i in cutlass.range_constexpr(len(cols)):
        vals = []
        for j in cutlass.range_constexpr(vec):
            # The trait call is hoisted OUT of the predicate and the mask is a per-field select: a
            # trait reference inside a dynamic `if` leaks the python object into the IR flattener
            # ("encountered a user-defined Python object"). Reading an unwritten frag slot is
            # harmless -- the select discards it.
            col = _off(cols[i], j)
            x = trait.leaf(frag[j, i], col)
            if const_expr(exact):
                vals.append(x)
            else:
                ok = col < hi
                vals.append(tuple(x[f] if ok else ident[f] for f in range(nf)))
        # `vec_linear` folds the run as ONE CHAIN (v0 + v1 + ... + vN) instead of the stride-
        # doubling tree. That is a DIFFERENT association, so it leaves ATen's bit pattern -- the
        # knob exists to price what a cheaper-to-fold run would buy (measured: nothing).
        inner = (
            _linear_reduce(vals, op)
            if const_expr(vec_linear)
            else _reduce_vec(vals, vec, op)
        )
        if const_expr(merge_per_group):
            inner = merge_lanes(trait, inner, merge_tpr, asc=True)
        _streaming_push(tree, inner, i, max_depth, op)
    out = tree[0]
    if const_expr(not merge_per_group):
        out = merge_lanes(trait, out, merge_tpr, asc=True)
    return out


@cute.jit
def fold_itree_warp(
    trait,
    frag,
    tm: cutlass.Constexpr,
    max_depth: cutlass.Constexpr,
    lane,
    w,
    base_col=0,
    bound=None,
    warp_stride=None,
    vec_linear: cutlass.Constexpr = False,
):
    """The per-chunk arm: a TileMap's strided groups, one lane butterfly per load."""
    return fold_groups(
        trait,
        frag,
        [tm.col_base(lane, w, l, warp_stride) + base_col for l in range(tm.loads)],
        const_expr(tm.vec),
        Int32(const_expr(tm.N)) if bound is None else bound,
        max_depth,
        const_expr(tm.tpr),
        merge_per_group=True,
        exact=const_expr(tm.exact),
        vec_linear=vec_linear,
    )


# Buffers in the staged fold's software pipeline (see _fold_itree_smem). 1 = stage-then-fold;
# 2 would overlap the next tile's cp.async with this tile's fold.
#
# ONE, because overlapping LOSES. Depth 2 doubles smem per row (4.5 KB -> 9 KB), and with one warp
# per block that halves the blocks an SM can hold, which this kernel feels more than it feels the
# transfer latency. MEASURED device us, depth 1 vs 2:
#
#   (  65536,   1024)   40.9 / 40.9   (ntiles=1, so depth 2 never engages)
#   (  32768,   2048)   38.8 / 46.5
#   (  16384,   4096)   39.2 / 47.0
#   (   8192,   8192)   39.6 / 47.9
#
# Same mechanism that sank the untiled version, where smem was the whole batch: on this fold, smem
# footprint buys occupancy and occupancy beats overlap.
_ITREE_STAGE_DEPTH = 1

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


@cute.jit
def fold_linear_rolled(
    trait,
    mX,
    r,
    vec: cutlass.Constexpr,
    nchunks,
    unroll: cutlass.Constexpr = _ROLL_UNROLL,
):
    """Fold row `r` with a runtime chunk loop. Returns an acc tuple. tpr == 1 only."""
    reduce_fn, acc_dt = trait.reduce, trait.acc
    acc = trait.init()
    gv = cute.flat_divide(mX[Int64(r), None], (vec,))
    frag = cute.make_rmem_tensor(cute.make_layout(vec), mX.element_type)
    for c in cutlass.range(nchunks, unroll=unroll):
        cute.autovec_copy(gv[None, c], frag)
        for i in cutlass.range_constexpr(vec):
            acc = reduce_fn(acc, acc_dt(frag[i]), c * Int32(vec) + Int32(i), True)
    return acc


@cute.jit
def _wide(rowv, base, vec: cutlass.Constexpr, frag_l):
    # `vec` elements (up to 128 bits) in ONE instruction. The (vec,) view is built by
    # pointer offset so `base` may be dynamic; frag_l is this load's slice of the fragment.
    src = cute.make_tensor(rowv.iterator + base, cute.make_layout(vec))
    cute.autovec_copy(src, frag_l)


@cute.jit
def load(
    mX,
    r,
    tm: cutlass.Constexpr,
    lane,
    w,
    frag,
    base_col=0,
    bound=None,
    warp_stride=None,
):
    """Fill `frag` ((vec, loads) rmem) with this thread's slice of row `r`.

    Emits ONE wide load per (thread, load) when the tile is exact. Otherwise the group is
    tested once -- a fully in-row group still takes the wide load, and only a ragged group
    falls back to per-element reads, with out-of-row elements left UNTOUCHED for the caller
    to treat as identity.
    """
    hi = Int32(const_expr(tm.N)) if bound is None else bound
    rowv = mX[Int64(r), None]
    for l in cutlass.range_constexpr(tm.loads):
        # base_col may be a python int (0, or a baked batch offset) or a DYNAMIC value
        # (the two-kernel stage 1 derives it from the block index); a plain add keeps a
        # static base static and promotes only when it has to.
        base = tm.col_base(lane, w, l, warp_stride) + base_col
        if const_expr(tm.exact and tm.wide_ok):
            _wide(rowv, base, tm.vec, frag[None, l])
        elif const_expr(not tm.wide_ok):
            for i in cutlass.range_constexpr(tm.vec):
                # `_off(base, i)` is inlined rather than bound to a name: binding a DYNAMIC
                # value inside a dynamic `if` is rejected ("None prior to this if, and update
                # to Int32 inside"). The compiler CSEs the repeated expression.
                if _off(base, i) < hi:
                    frag[i, l] = rowv[_off(base, i)]
        else:
            if _off(base, tm.vec) <= hi:
                _wide(rowv, base, tm.vec, frag[None, l])
            else:
                for i in cutlass.range_constexpr(tm.vec):
                    if _off(base, i) < hi:
                        frag[i, l] = rowv[_off(base, i)]


def make_fragment(mX, tm) -> cute.Tensor:
    return cute.make_rmem_tensor(cute.make_layout((tm.vec, tm.loads)), mX.element_type)


def smem_box_layout(N: int, threads: int):
    """Smem destination for a (threads, N) TMA box of whole rows: plain row-major.

    The bank conflict a whole-row read implies is dealt with in the ACCESS PATTERN (see
    fold_smem_rotated), not here, because neither layout-side fix works -- both were built
    and measured:
      * an arbitrary XOR swizzle (row bits into the bank bits) is REJECTED by the atom
        ("unable to partition input tensors for TMA"); TMA supports only the GEMM swizzle
        family (32B/64B/128B), whose phase pattern does not de-conflict a whole-row read.
      * a transposed (column-major) destination BUILDS AND RUNS BUT IS WRONG -- the transfer
        cannot transpose, so logical (t, c) indexing then reads the wrong elements.
    """
    return cute.make_ordered_layout((threads, N), order=(1, 0))


@cute.jit
def fold_smem_rotated(trait, sX, rb, N: cutlass.Constexpr):
    """Fold row `rb` of a staged (threads, N) smem tile. One thread per row, no lane merge.

    Indexed LOGICALLY so the true column reaches the trait -- reading `vec` physically
    adjacent words instead would hand an index trait (argmax) a permuted position.

    ROTATE each thread's read order by its row index: at step c, thread t reads column
    (c + t) % N. Thread t reads row t, so an unrotated row-major read puts every lane of a
    warp in bank c % 32 -- a 32-way conflict, measured to cost MORE than the coalescing the
    TMA staging buys (0.86-0.91x, a regression). The rotation is a swizzle in the ACCESS
    PATTERN, which needs no layout support at all. Legal because this path carries a numeric
    contract, not a bitwise one, and the true column still reaches the trait.

    N is a power of two here (see kernel_rowtile.tma_ok), so the modulo is a mask.
    """
    acc = trait.init()
    mask = const_expr(N - 1)
    for c in cutlass.range_constexpr(N):
        col = (Int32(c) + rb) & Int32(mask)
        acc = trait.reduce(acc, trait.acc(sX[rb, col]), col, True)
    return acc


@cute.jit
def fold_cols_rolled(
    trait,
    mX,
    col,
    row0,
    nrows,
    vec: cutlass.Constexpr,
    unroll: cutlass.Constexpr = _ROLL_UNROLL,
):
    """Accumulate DOWN the rows, keeping `vec` independent accumulators. For columns.

    The transpose of every other fold here. A row reduction collapses a thread's fragment to
    ONE accumulator and then merges lanes; a column reduction vectorizes along the CONTIGUOUS
    (kept) axis, so each thread owns `vec` adjacent columns, carries one accumulator per
    column, and never merges across lanes at all. The addressing is still a per-row wide load,
    which is what makes it reachable from these same primitives.

    State is a loop-carried TUPLE of acc tuples, one per column this thread owns.
    """
    reduce_fn, acc_dt = trait.reduce, trait.acc
    accs = tuple(trait.init() for _ in range(vec))
    frag = cute.make_rmem_tensor(cute.make_layout(vec), mX.element_type)
    for r in cutlass.range(nrows, unroll=unroll):
        # row0 offsets this thread's slice of the REDUCED axis, which is what lets the driver
        # split that axis P ways and hand each block its own chunk of rows.
        rr = row0 + Int32(r)
        cute.autovec_copy(
            cute.flat_divide(mX[Int64(rr), None], (vec,))[None, col], frag
        )
        # plain `range`: a comprehension is not visited by the DSL AST preprocessor, so
        # range_constexpr raises there -- and vec is compile-time anyway, so this unrolls
        # at trace time exactly like a range_constexpr `for` statement would.
        accs = tuple(reduce_fn(accs[i], acc_dt(frag[i]), rr, True) for i in range(vec))
    return accs


class TileReduce:
    """The tile reduction KERNEL: one body, parameterized by which axis is reduced.

    axis "row" -- the reduced axis is CONTIGUOUS. `tpr` threads share a row, each folding its
        own chunks of it, and then the lanes merge (and the warps too, when a row spans more
        than one). tpr == 1 is the narrow-row shape: one thread owns the whole row and nothing
        merges. `use_tma` stages the block's rows through a TMA box first.
    axis "col" -- the reduced axis is STRIDED (dim 0). Each thread owns `vec` ADJACENT columns
        and folds DOWN the rows, so there is one accumulator per output and nothing merges
        across lanes at all. The y-grid splits the reduced axis, and `combine` folds the
        partials that split leaves, in a second pass of this same body.
    axis "general" -- ANY layout. One block per output, all `nt` threads on it, addressing by
        TensorIterator-derived mixed-radix decode (fold_decoded), so a transposed, sliced,
        permuted, expanded or overlapping-window input needs no reshape and no special case.
        This is the arm that makes "never decline a geometry" possible, and it is also the
        COMBINE engine for every split whose partials are laid out per output (`combine`).

    `order` is orthogonal to `axis` and applies to the row axis: "linear" is the default
    rolled fold, "inner_tree" is a COMPILE-TIME add DAG (a stride-doubling tree per load plus a
    streaming carry) whose bit pattern is reproducible and pinnable. It brings its own thread
    map -- (lane, warp) rather than (lane, row) -- and its own cross-warp step, an ascending
    butterfly over smem rather than the generic block reduce, because both are part of the DAG
    and neither can be swapped for the shared version without changing the bits. Its plan
    (per-batch tiles, warps per row, carry depth) is the driver's, in `itree`, and the plan's
    SHAPE also picks the thread map: one thread per row for a row that fits one fragment or for
    the per-row combine, `wpr` warps per row otherwise.

    Why one body and not three: only the FOLD is axis-specific, and every fold is a
    primitive above. The general axis runs at tpr == nt (every thread of the block on one
    output), which is exactly the row axis's tail -- merge_lanes clamps to a warp and
    block_reduce defaults to one row per block -- so it inherits the lane merge, the warp
    merge unchanged. What follows it is shared by all three: the clamp of dead threads, the
    projection (which has to happen OUTSIDE the store branch either way, or the DSL rejects
    the binding) and the store -- every axis writes nslots x nouts results and they differ
    only in the index written to. The col axis pins
    `lane = 0`, since its mapping already gives every output group a thread of its own; that
    is what lets one store serve both.
    """

    def __init__(
        self,
        trait,
        dtype,
        axis,
        N,
        tpr=1,
        nt=128,
        nouts=1,
        final=True,
        unroll=_ROLL_UNROLL,
        vec=None,
        use_tma=False,
        combine=False,
        pc=True,
        npairs_red=0,
        npairs_kept=0,
        gidx_from="r",
        flat_tail=False,
        ragged_chunk=False,
        order="linear",
        # duck-typed like `trait`: the plan's type belongs to the DRIVER above
        # this module (kernel_rowtile._ItreePlan), so naming it here would invert
        # the dependency.
        itree: Any = None,
    ):
        if axis not in ("row", "col", "general"):
            raise ValueError(f"axis must be 'row', 'col' or 'general', got {axis!r}")
        if order not in ("linear", "inner_tree"):
            raise ValueError(f"order must be 'linear' or 'inner_tree', got {order!r}")
        if order == "inner_tree":
            if axis != "row" or itree is None:
                raise ValueError(
                    "the inner-tree order is a row-axis option and needs a plan"
                )
            # Its thread map IS its plan: `wpr` chunks per row (0 for the one-thread-per-row
            # shapes) folded by `wpr // kchunk` warps, and `rows_per_block` rows per block.
            tpr = WARP * (itree.wpr // itree.kchunk) if itree.wpr else 1
            if itree.stage_e:
                tpr = WARP
            nt = tpr * itree.rows_per_block
        if axis == "general":
            # Every thread of the block folds the one output, so the tail is the row axis's
            # at tpr == nt.
            tpr = nt
            if nt % WARP:
                # _block_merge derives warps_per_row = nt // WARP, so a block that is not a
                # whole number of warps leaves the last partial warp's accumulator OUT of
                # the merge -- a silently wrong reduction (measured 12-62% low), not a short
                # buffer. `block` is caller-settable on reduce_dim / reduce_all, so the row
                # axis's check below is not enough on its own.
                raise ValueError(f"a general-axis block must be whole warps, got {nt=}")
        if axis == "row" and tpr != 1 and (tpr % WARP or tpr > nt or nt % tpr):
            raise ValueError(
                f"tpr must be 1 or a multiple of {WARP} dividing nt: {tpr=} {nt=}"
            )
        if use_tma and (axis != "row" or tpr != 1):
            raise ValueError(
                f"TMA stages whole rows: needs row at tpr 1, {axis=} {tpr=}"
            )
        if use_tma and N & (N - 1):
            # fold_smem_rotated rotates with `& (N - 1)`, which is a rotation only at a
            # power-of-two N: at any other N it duplicates some columns and skips others,
            # so the fold runs over the wrong multiset and returns a plausible number.
            # tma_ok declines those N, but use_tma is caller-settable and bypasses it.
            raise ValueError(f"TMA staging needs a power-of-two row length, got {N=}")
        self.trait = trait
        self.dtype = dtype
        self.axis = axis
        self.N = N
        self.tpr = tpr
        self.nt = nt
        self.nouts = nouts
        self.final = final
        self.unroll = unroll
        self.use_tma = use_tma
        self.combine = combine
        self.pc = pc  # partial layout: (P, C) when True, else (C, P)
        # general-axis addressing policy, all compile-time (see fold_decoded)
        self.npairs_red = npairs_red
        self.npairs_kept = npairs_kept
        self.gidx_from = gidx_from
        self.flat_tail = flat_tail
        self.ragged_chunk = ragged_chunk
        self.order = order
        self.itree = itree
        itemsize = dtype.width // 8 if dtype is not None else 0
        # The row folds take their tile from TileMap; loads=1 because they are ROLLED, so the
        # static per-thread count is unused and the MAX_UNROLL bound is trivially met. The TMA
        # fold is the exception -- it walks the staged row with a compile-time trip count, so
        # declare its real depth and let the bound apply. The col axis needs no tile: its
        # `vec` is a driver choice (accumulators per thread, not just load width).
        # The inner-tree order carries a tile PER BATCH in its plan, so it needs none here.
        self.tm = (
            TileMap(N, itemsize, tpr, N // vec_size(N, itemsize) if use_tma else 1)
            if axis == "row" and order == "linear"
            else None
        )
        # the general axis loads one element at a time (an arbitrary stride pattern has no
        # width to exploit), so it has no tile and no vector
        if axis == "row":
            self.vec = self.tilemap.vec if order == "linear" else itree.vec
        else:
            self.vec = 1 if axis == "general" else vec
        # one output per thread on the row axis (its lanes are merged first); `vec` adjacent
        # columns, each with its own accumulator, on the col axis
        self.nslots = self.vec if axis == "col" else 1
        self.rows_per_block = nt // tpr
        self.warps_per_row = tpr // WARP  # 0 at tpr == 1: nothing to merge
        self.tiler = (nt, N)  # TMA box: nt whole rows

    @property
    def tilemap(self):
        # Only a LINEAR row fold carries one tile: the inner-tree order keeps a tile per batch
        # in its plan, and the col/general axes have none at all.
        if self.tm is None:
            raise AssertionError(f"no tile on the {self.axis} axis")
        return self.tm

    @property
    def cache_sig(self):
        # N is ABSENT except for the TMA variant, whose box shape is compile-time: every other
        # path takes its extents at runtime, so one compiled kernel serves a whole vec class.
        return (
            self.axis,
            self.vec,
            self.tpr,
            self.nt,
            self.nouts,
            self.final,
            self.unroll,
            self.use_tma,
            self.combine,
            self.pc,
            self.trait.nfields,
            self.N if self.use_tma else 0,
            self.npairs_red,
            self.npairs_kept,
            self.gidx_from,
            self.flat_tail,
            self.ragged_chunk,
            self.order,
            # A fixed DAG is a compile-time object, so unlike every other arm this one keys on
            # N (through the plan). That is the cost of a reproducible bit pattern: one kernel
            # per shape rather than one per vec class.
            self.itree.sig if self.itree is not None else None,
        )

    @cute.jit
    def _fold_tma(self, mX, tma_atom, bx, tx):
        # Stage this block's (nt, N) box of WHOLE rows into smem with one descriptor-driven
        # transfer, then fold row tx out of smem. See kernel_rowtile._TMA_MIN_STRIDE for why,
        # and fold_smem_rotated for the bank rotation the fold needs.
        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            self.dtype, smem_box_layout(self.N, self.nt), byte_alignment=16
        )
        mbar = smem.allocate_array(cutlass.Int64, num_elems=2)
        gX = cute.local_tile(mX, self.tiler, (cutlass.Int64(bx), 0))
        pipe = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, const_expr(self.nt)
            ),
            tx_count=const_expr(cute.size(self.tiler) * self.dtype.width // 8),
            barrier_storage=mbar,
        )
        pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        tSsX, tSgX = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(gX, 0, 2),
        )
        # Warp 0 issues the transfer. Narrowing this to a single THREAD deadlocks (the GPU
        # spins at 100%): PipelineTmaAsync signals per warp-lane-0 and syncs the warp
        # internally, so the producer region must be entered by a whole warp even though the
        # producer group is size 1. Rows of the last tile past M are zero-filled by the
        # descriptor; those threads' accumulators are discarded at the guarded store.
        if cute.arch.warp_idx() == 0:
            pipe.producer_acquire(pstate)
            cute.copy(
                tma_atom, tSgX, tSsX, tma_bar_ptr=pipe.producer_get_barrier(pstate)
            )
            pipe.producer_commit(pstate)
        pipe.consumer_wait(cstate)
        acc = fold_smem_rotated(self.trait, sX, tx, const_expr(self.N))
        pipe.consumer_release(cstate)
        return acc

    @cute.jit
    def _block_merge(self, acc):
        # Merge the per-warp accumulators of one output through smem. A no-op unless the
        # output spans more than one warp, which is every general-axis launch of >= 64
        # threads and any row shape with tpr > WARP.
        if const_expr(self.warps_per_row <= 1):
            return acc
        from .._cutedsl.traits import block_reduce

        trait = self.trait
        smem = cutlass.utils.SmemAllocator()
        bufs = [
            smem.allocate_tensor(
                trait.fdtypes[f],
                cute.make_layout(self.rows_per_block * self.warps_per_row),
                byte_alignment=8,
            )
            for f in range(trait.nfields)
        ]
        return block_reduce(
            trait,
            acc,
            bufs,
            const_expr(self.warps_per_row),
            const_expr(self.rows_per_block),
        )

    @cute.jit
    def _fold_itree(self, mX, r, lane_w, warp_id, row_in_block, batch_idx=None):
        """The inner-tree fold: a static fragment per batch, tree-folded, carried linearly.

        The row is covered in compile-time BATCHES (a fragment cannot hold an arbitrary row),
        each with its own tile; within a batch the fold is the stride-doubling tree plus the
        streaming carry, and the per-warp results meet in smem for one ASCENDING butterfly.
        Batches then accumulate LINEARLY, outside the tree. Every one of those choices is part
        of the DAG this order exists to reproduce.

        The SPLIT shape has one batch per BLOCK instead, indexed by `batch_idx` at runtime: its
        width and per-warp chunk are selected here rather than baked, and the one compiled tile
        (the full batch's) reaches fewer loads in the short last batch through a per-warp bound.
        """
        trait = self.trait
        it = self.itree
        nf = const_expr(trait.nfields)
        op, ident = leaf_op(trait), identity(trait)
        warp_writes = []  # empty when one warp covers the row: nothing to stage
        if const_expr(it.wpr // max(it.kchunk, 1) > 1):
            # One staging buffer PER FIELD: an index trait's position and a Welford count are
            # different dtypes, so they cannot share a tensor.
            smem = cutlass.utils.SmemAllocator()
            warp_writes = [
                smem.allocate_tensor(
                    trait.fdtypes[f],
                    cute.make_layout(
                        const_expr((it.wpr // it.kchunk) * it.rows_per_block)
                    ),
                    byte_alignment=8,
                )
                for f in range(nf)
            ]
        # Only the looped shape SEEDS the cross-batch accumulator with the identity, exactly as
        # upstream does. The single-batch shapes return their batch's own accumulator, and that
        # is not the same value: `0.0 + -0.0` is `+0.0`.
        final = None
        kc = const_expr(it.kchunk)  # ADJACENT CHUNKS this thread group folds
        groups = const_expr(it.wpr // it.kchunk) if it.wpr else 0
        for b in cutlass.range_constexpr(len(it.tms)):
            tm = it.tms[b]
            # Issue EVERY chunk's loads before folding any of them: a chunk is only `loads` deep
            # (one load at the widths ATen's plan picks below N=2048), so folding chunk by chunk
            # would leave a single load in flight and the thread would stall on latency.
            frags, bases, bounds, wstrides = [], [], [], []
            for c in cutlass.range_constexpr(kc):
                cid = warp_id * Int32(kc) + Int32(c) if const_expr(kc > 1) else warp_id
                if const_expr(it.shape == "split"):
                    nbatch, bte, last, ch_full, ch_last = it.split
                    is_last = batch_idx == Int32(nbatch - 1)
                    rem = Int32(last) if is_last else Int32(bte)
                    chunk = Int32(ch_last) if is_last else Int32(ch_full)
                    warp_off = cid * chunk
                    warp_off = rem if warp_off > rem else warp_off  # noqa: FURB136 -- no min
                    base = batch_idx * Int32(bte) + warp_off
                    tail = rem - warp_off
                    # The chunk's own end, which is what stops a chunk shorter than the baked load
                    # count from reading the NEXT one's elements. ATen zeroes whole loads past its
                    # share; a bound at the same place is the same masking.
                    bound = base + (chunk if chunk < tail else tail)  # noqa: FURB136 -- no min
                    wstride = Int32(0)  # the chunk offset is already in `base`
                elif const_expr(kc > 1):
                    # Fold the chunk offset into the base so one tile serves every chunk.
                    base = Int32(const_expr(it.batches[b][0])) + cid * Int32(
                        const_expr(tm.loads * WARP * tm.vec)
                    )
                    bound, wstride = None, Int32(0)
                else:
                    base = const_expr(it.batches[b][0])
                    bound, wstride = None, None
                frag = make_fragment(mX, tm)
                load(mX, r, tm, lane_w, cid, frag, base, bound, wstride)
                frags.append(frag)
                bases.append(base)
                bounds.append(bound)
                wstrides.append(wstride)
            # Each chunk's own tree, then the balanced tree over this thread's k ADJACENT chunks.
            # That local combine is bit-exactly what the cross-chunk merge would have done for
            # them, so fusing chunks moves work off shuffles/smem without moving the DAG -- and
            # unlike fusing adjacent VECTORS it leaves every load's lanes `vec` apart, i.e. still
            # perfectly coalesced.
            accs = [
                fold_itree_warp(
                    trait,
                    frags[c],
                    tm,
                    const_expr(it.depth),
                    lane_w,
                    warp_id * Int32(kc) + Int32(c) if const_expr(kc > 1) else warp_id,
                    bases[c],
                    bounds[c],
                    wstrides[c],
                    const_expr(it.vec_linear),
                )
                for c in range(kc)
            ]
            warp_acc = _inner_tree_reduce(accs, op)
            if const_expr(groups > 1):
                slot = row_in_block * Int32(groups)
                if lane_w == Int32(0):
                    for f in cutlass.range_constexpr(nf):
                        warp_writes[f][slot + warp_id] = warp_acc[f]
                cute.arch.barrier()
                # Read every field unconditionally and mask with a select, but CLAMP the slot: a
                # warp has 32 lanes and the buffer only holds `groups` per row, so an unclamped
                # read runs off the end. A dead lane reads slot 0 and the select discards it.
                live = lane_w < Int32(groups)
                idx = slot + (lane_w if live else Int32(0))
                got = tuple(warp_writes[f][idx] for f in range(nf))
                merged = tuple(got[f] if live else ident[f] for f in range(nf))
                warp_acc = merge_lanes(trait, merged, const_expr(tm.tpr), asc=True)
                if const_expr(b + 1 < len(it.tms)):
                    cute.arch.barrier()
            if const_expr(it.shape == "looped" and b == 0):
                final = op(ident, warp_acc)
            elif const_expr(b == 0):
                final = warp_acc
            else:
                final = op(final, warp_acc)
        return final

    @cute.jit
    def _fold_itree_smem(self, mX, r, lane, row_in_block):
        """Stage through smem in TILES, so one warp folds a batch with `span/T` butterflies while
        smem stays at T columns per row regardless of how wide the batch is.

        The butterfly count is span/(WARP*vec) -- one per 32-lane load -- because a coalesced load
        leaves a lane owning only `vec` adjacent columns, and in-register tree levels need an
        ALIGNED CONTIGUOUS run. Staging breaks the tie: the global read stays coalesced, the lane
        reads its own contiguous run back out of smem, and one butterfly covers the whole tile.
        Tiles then combine through the streaming carry, so the DAG is still the balanced tree over
        the batch's columns and the bits do not move.
        `T = stage_e * WARP` is the knob: T = span is one butterfly and smem the whole batch, while
        a smaller T trades butterflies back for a smem footprint that no longer grows with N.

        PIPELINED at `_ITREE_STAGE_DEPTH` buffers: at depth 2 the next tile's transfer is issued
        BEFORE the current tile is folded, so the copy overlaps the arithmetic instead of the warp
        stalling on `cp_async_wait_group(0)` between them. Costs one more tile of smem.

        SMEM LAYOUT is (stage_e, WARP) per buffer with the lane stride padded by `vec`, i.e. tile
        column c at `c + vec * (c // stage_e)`. The padding is what makes both directions
        conflict-free: a lane's run stays contiguous (both the store and the load stay 16-byte
        wide) while the run starts land `vec` words apart, so 32 lanes hit 32 distinct bank groups
        instead of one.
        """
        trait = self.trait
        it = self.itree
        # The leaf/mask/vec-tree work lives in fold_groups now; this body only needs the combiner
        # for the cross-tile carry and the identity for the seed.
        op, ident = leaf_op(trait), identity(trait)
        Es = const_expr(it.stage_e)  # columns per lane PER TILE
        vec = const_expr(it.vec)
        span = const_expr(it.batches[0][2] * it.wpr * WARP * it.vec)
        wle = const_expr(WARP * vec)
        tile_cols = const_expr(Es * WARP)
        ntiles = const_expr(span // tile_cols)
        pitch = const_expr(Es + vec)
        stride = const_expr(pitch * WARP)  # one buffer, one row
        depth = const_expr(min(_ITREE_STAGE_DEPTH, ntiles))
        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            self.dtype,
            cute.make_layout(const_expr(stride * depth * it.rows_per_block)),
            byte_alignment=16,
        )
        base = row_in_block * Int32(const_expr(stride * depth))
        rowv = mX[Int64(r), None]
        hi = Int32(const_expr(it.batches[0][1]))  # this batch's real column count
        gv = cute.flat_divide(rowv, (vec,))
        sv = cute.flat_divide(sX, (vec,))
        g2s = cute.make_copy_atom(
            cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=const_expr(vec * mX.element_type.width),
        )
        runfrag = cute.make_rmem_tensor(
            cute.make_layout((vec, const_expr(Es // vec))), mX.element_type
        )

        # cp.async goes GLOBAL -> SMEM directly, so the data never lands in registers: one
        # instruction per vec group instead of a load plus a store. The atom needs a statically
        # 16-byte-aligned source AND destination, which a raw `iterator + offset` does not carry --
        # flat_divide views inherit the wrap's alignment.
        for pre in cutlass.range_constexpr(depth):
            for i in cutlass.range_constexpr(tile_cols // wle):
                off = const_expr(pre * tile_cols + i * wle)
                col = Int32(const_expr(off)) + lane * Int32(vec)
                k = Int32(const_expr(off // vec)) + lane
                ks = k if col + Int32(vec) <= hi else Int32(0)  # clamp a short batch
                local = Int32(const_expr(i * wle)) + lane * Int32(vec)
                dst = (
                    base
                    + Int32(const_expr((pre % depth) * stride))
                    + (local // Int32(Es)) * Int32(pitch)
                    + local % Int32(Es)
                )
                cute.copy(g2s, gv[None, ks], sv[None, dst // Int32(vec)])
            cute.arch.cp_async_commit_group()

        tree: list = []
        for t in cutlass.range_constexpr(ntiles):
            # Only THIS tile's transfer has to have landed; the ones for tiles t+1..t+depth-1 stay
            # in flight, which is where the overlap comes from.
            cute.arch.cp_async_wait_group(const_expr(min(depth - 1, ntiles - 1 - t)))
            cute.arch.sync_warp()

            # FOLD this lane's own contiguous run of the tile, then ONE butterfly for the tile.
            run = base + Int32(const_expr((t % depth) * stride)) + lane * Int32(pitch)
            col0 = Int32(const_expr(t * tile_cols)) + lane * Int32(Es)
            # Read the whole run into registers first, so the SHARED fold body applies: the same
            # (fragment, columns) interface as the per-chunk arm. Only the columns differ -- they
            # are contiguous here -- and the lane merge happens once at the end.
            for i in cutlass.range_constexpr(Es // vec):
                cute.autovec_copy(
                    cute.make_tensor(
                        sX.iterator + run + Int32(const_expr(i * vec)),
                        cute.make_layout(vec),
                    ),
                    runfrag[None, i],
                )
            _streaming_push(
                tree,
                fold_groups(
                    trait,
                    runfrag,
                    [col0 + Int32(const_expr(i * vec)) for i in range(Es // vec)],
                    vec,
                    hi,
                    const_expr(_ilog2(Es // vec)),
                    WARP,
                    merge_per_group=False,
                ),
                t,
                const_expr(_ilog2(max(ntiles, 2))),
                op,
            )
            if const_expr(t + depth < ntiles):
                # This buffer is free now, so refill it for the tile `depth` ahead. It MUST come
                # after the fold: issuing it earlier is a write-after-read race on the buffer we
                # are reading, which shows up as wrong results only once M is large enough for the
                # transfer to actually land mid-fold.
                cute.arch.sync_warp()
                nxt = const_expr(t + depth)
                for i in cutlass.range_constexpr(tile_cols // wle):
                    off = const_expr(nxt * tile_cols + i * wle)
                    col = Int32(const_expr(off)) + lane * Int32(vec)
                    k = Int32(const_expr(off // vec)) + lane
                    ks = k if col + Int32(vec) <= hi else Int32(0)
                    local = Int32(const_expr(i * wle)) + lane * Int32(vec)
                    dst = (
                        base
                        + Int32(const_expr((nxt % depth) * stride))
                        + (local // Int32(Es)) * Int32(pitch)
                        + local % Int32(Es)
                    )
                    cute.copy(g2s, gv[None, ks], sv[None, dst // Int32(vec)])
                cute.arch.cp_async_commit_group()
        # SEED with the identity, exactly as the looped shape does: this path only ever serves a
        # single batch, and `ident + x` is NOT a no-op for a signed zero -- dropping it returns
        # -0.0 where ATen returns +0.0.
        return op(ident, tree[0])

    @cute.jit
    def _fold_itree_combine(self, mIns, row):
        """Stage 2 of the split shape: fold one row's partials LINEARLY, ascending.

        Starts at partial 0 rather than the identity (see `_fold_itree`), and takes its trip
        count as a RUNTIME loop: the batch count grows with N, so unrolling it would make
        compile time scale with the reduction. One buffer per trait field.
        """
        # Bind the trait's methods to locals: attribute access on it inside a dynamic loop trips
        # the IR flattener ("encountered a user-defined Python object").
        combine_fn, fdtypes = self.trait.combine, self.trait.fdtypes
        nf = const_expr(self.trait.nfields)
        nbatch = const_expr(self.itree.split[0])
        base = row * Int32(nbatch)
        pull = lambda i: tuple(fdtypes[f](mIns[f][i]) for f in range(nf))  # noqa: E731
        acc = pull(base)
        for b in cutlass.range(1, nbatch):
            acc = combine_fn(acc, pull(base + b))
        return acc

    @cute.jit
    def _fold_partials(self, mIns, unit, nchunks, npar):
        # COMBINE pass (col axis stage 2): fold the npar partials of this thread's column,
        # which the split left as one (npar, nchunks) matrix per trait field. Bind the trait's
        # methods to locals -- attribute access on it inside a dynamic loop trips the IR
        # flattener.
        trait = self.trait
        combine_fn = trait.combine
        fdtypes = trait.fdtypes
        nf = const_expr(trait.nfields)
        acc = trait.init()
        for pp in cutlass.range(npar, unroll=const_expr(self.unroll)):
            base = Int32(pp) * nchunks + unit
            acc = combine_fn(acc, tuple(fdtypes[f](mIns[f][base]) for f in range(nf)))
        return acc

    @cute.jit
    def __call__(
        self,
        mIns: list,
        mOuts: list,
        nchunks,
        nwaves,
        project_n,
        q,
        npar,
        rvals,
        kvals,
        in_base,
        limit,
        stream,
    ):
        # The TMA atom is built here (host-compile time, inside the jit region) and baked into
        # the kernel; the input is replaced by the descriptor tensor for the load.
        if const_expr(self.use_tma):
            tma_atom, mTma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mIns[0],
                smem_box_layout(self.N, self.nt),
                self.tiler,
            )
            mIns = [mTma]
        else:
            tma_atom = None
        if const_expr(self.axis == "general"):
            # one block per output, read live so one compile serves any output count
            gx = mOuts[0].shape[0]
            gy = Int32(1)
        elif const_expr(self.axis == "row"):
            # Count OUTPUTS under the inner-tree order: its split shape writes one partial per
            # (row, batch), so the input's row count is not the block count.
            nout = (
                mOuts[0].shape[0]
                if const_expr(self.order == "inner_tree")
                else mIns[0].shape[0]
            )
            gx = cute.ceil_div(nout, const_expr(self.rows_per_block))
            gy = Int32(1)
        else:
            gx = cute.ceil_div(nchunks, const_expr(self.nt))
            # the y-grid splits the REDUCED axis in stage 1; combine has already consumed it
            gy = Int32(1) if const_expr(self.combine) else npar
        self.kernel(
            mIns,
            mOuts,
            tma_atom,
            nchunks,
            nwaves,
            project_n,
            q,
            npar,
            rvals,
            kvals,
            in_base,
            limit,
        ).launch(grid=[gx, gy, 1], block=[const_expr(self.nt), 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mIns: list,
        mOuts: list,
        tma_atom,
        nchunks,
        nwaves,
        project_n,
        q,
        npar,
        rvals,
        kvals,
        in_base,
        limit,
    ):
        # RUNTIME args, so one compiled kernel serves every extent sharing a structure:
        #   nchunks   vec-chunks along the axis a thread walks (row: of its row; col: of the
        #             column count, i.e. how many threads have work)
        #   nwaves    row only: waves of tpr chunks the rolled fold takes
        #   project_n the TRUE reduced extent -- mean/var's divisor, and on the col axis the
        #             row count its per-block chunk bound is measured against
        #   q, npar   col only: the reduced-axis split (rows per chunk, chunk count)
        #   rvals     general only: the reduced (extent, stride) magic quads to decode
        #   kvals     general only: the same for the kept dims, decoding this block's output
        #   in_base   general only: flat input offset of output coordinate 0
        #   limit     general only: the bound the fold clamp measures against
        #             (nchunks doubles as the general axis's per-output step count)
        #
        # An arg a variant does not use is passed as None, NOT as a dummy value: one extra
        # unused Int32 param measured 1.27x on the column fold (8.2 -> 10.4us at (65536, 256)
        # fp32 sum), which is why the drivers pin the other axis's args to None.
        tx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        trait = self.trait
        chunk_base = Int32(0)  # nonzero only under ragged_chunk, for gidx_from "chunk"
        batch_idx = None  # split shape only: which batch of the row this block folds
        # inner-tree order only: its (lane, warp) thread map, set from the plan below
        lane_w = warp_id = row_in_block = None
        if const_expr(self.order == "inner_tree"):
            # (lane, warp) rather than (lane, row): `wpr` warps cooperate on one row, and
            # `rows_per_block` such groups share the block.
            wpr = const_expr(self.itree.wpr)
            if const_expr(wpr == 0):
                # One thread per row (the multirow fold and the split's combine): no lane to
                # merge with, so every thread holds its own total and stores it.
                lane_w, warp_id, row_in_block, lane = (Int32(0),) * 4
            elif const_expr(self.itree.stage_e):
                # SMEM-STAGED: exactly one warp per row, so there is no chunk index and no
                # cross-warp merge -- the whole batch lands in one butterfly.
                lane_w = Int32(tx % WARP)
                lane = lane_w
                warp_id = Int32(0)
                row_in_block = Int32(tx // WARP)
            else:
                # `warp_id` is the thread group, i.e. which BLOCK of kchunk adjacent chunks
                # this warp folds -- the fold multiplies it up to real chunk indices.
                groups = const_expr(wpr // self.itree.kchunk)
                lane_w = Int32(tx % WARP)
                warp = Int32(tx // WARP)
                row_in_block = warp // Int32(groups)
                warp_id = warp % Int32(groups)
                # 0 for exactly the one thread per row that holds the merged total, which is
                # what the shared store's `lane == 0` guard wants.
                lane = Int32(tx % const_expr(WARP * groups))
            if const_expr(self.itree.shape == "split"):
                # The grid pairs every row with every batch, so the block index carries both --
                # and the partial this block writes is at that same index.
                nb = const_expr(self.itree.split[0])
                raw = Int32(bx) // Int32(nb)
                batch_idx = Int32(bx) % Int32(nb)
                alive = True
            elif const_expr(wpr == 0):
                raw = Int32(bx) * const_expr(self.nt) + Int32(tx)
                alive = raw < Int32(mOuts[0].shape[0])
            else:
                raw = Int32(bx) * const_expr(self.itree.rows_per_block) + row_in_block
                alive = raw < Int32(mOuts[0].shape[0])
        elif const_expr(self.axis == "general"):
            # One block per output: the block index IS the output, and every thread folds it.
            raw = Int32(bx)
            lane = Int32(tx)
            alive = True
        elif const_expr(self.axis == "row"):
            raw = Int32(bx) * const_expr(self.rows_per_block) + Int32(
                tx // const_expr(self.tpr)
            )
            lane = Int32(tx % const_expr(self.tpr))
            alive = raw < Int32(mIns[0].shape[0])
        else:
            raw = Int32(bx) * const_expr(self.nt) + Int32(tx)
            lane = Int32(0)  # the col mapping gives every output group its own thread
            alive = raw < nchunks
        # Dead threads clamp onto unit 0 so every load stays in range; their accumulator is
        # computed and discarded at the guarded store. Cheaper than predicating every load,
        # and a fold has no side effects.
        unit = raw if alive else Int32(0)

        # THE FOLD: the one axis-specific step. Everything after it is shared.
        if const_expr(self.order == "inner_tree"):
            if const_expr(self.itree.shape == "combine"):
                accs = (self._fold_itree_combine(mIns, unit),)
            elif const_expr(self.itree.stage_e):
                accs = (self._fold_itree_smem(mIns[0], unit, lane_w, row_in_block),)
            else:
                accs = (
                    self._fold_itree(
                        mIns[0], unit, lane_w, warp_id, row_in_block, batch_idx
                    ),
                )
        elif const_expr(self.axis == "general"):
            # Base flat offset of this output (decode the block index against the kept dims;
            # zero kept pairs -- a full reduction -- leaves just in_base), then the fold bound.
            obase = in_base
            if const_expr(self.npairs_kept > 0):
                obase = in_base + _decode_offset(
                    unit, kvals, const_expr(self.npairs_kept)
                )
            rb = nchunks
            if const_expr(self.flat_tail):
                # Reduce-all stage 1: the last chunk overhangs the flat input, so clamp to
                # what is left before `limit`.
                left = limit - obase
                c64 = cutlass.Int64(nchunks)
                left = left if left < c64 else c64  # noqa: FURB136 -- no DSL builtin min
                zero = cutlass.Int64(0)
                left = left if left > zero else zero  # noqa: FURB136 -- no builtin max
                rb = cutlass.Int32(left)
            elif const_expr(self.ragged_chunk):
                # A split whose chunk need not divide the reduced run: the LAST chunk of every
                # output is short and must fold nothing belonging to the next one. `limit`
                # carries the reduced extent; the chunk pair is the fastest-varying kept pair,
                # so its magic quad yields the chunk index with no runtime divide -- once per
                # BLOCK, not per element. Counted in STEPS, so it is independent of the
                # reduced axis's stride (a contiguous row split and a column split both use
                # it unchanged).
                qq = (cutlass.Int64(unit) * kvals[0]) >> kvals[1]
                c = cutlass.Int64(unit) - qq * kvals[2]
                cnt = cutlass.Int64(nchunks)
                chunk_base = Int32(c * cnt)  # this chunk's first step, for gidx
                left = limit - c * cnt
                c64 = cutlass.Int64(nchunks)
                left = left if left < c64 else c64  # noqa: FURB136 -- no builtin min
                zero = cutlass.Int64(0)
                left = left if left > zero else zero  # noqa: FURB136 -- no builtin max
                rb = cutlass.Int32(left)
            if const_expr(self.combine):
                accs = (
                    fold_partials_run(
                        trait, mIns, obase, rb, const_expr(self.nt), lane, in_base
                    ),
                )
            else:
                accs = (
                    fold_decoded(
                        trait,
                        mIns[0],
                        obase,
                        rvals,
                        const_expr(self.npairs_red),
                        rb,
                        const_expr(self.nt),
                        lane,
                        in_base,
                        chunk_base,
                        const_expr(self.gidx_from),
                    ),
                )
            accs = (merge_lanes(trait, accs[0], const_expr(self.tpr)),)
            accs = (self._block_merge(accs[0]),)
        elif const_expr(self.combine):
            accs = (self._fold_partials(mIns, unit, nchunks, npar),)
        elif const_expr(self.axis == "col"):
            # This block's chunk of the REDUCED axis. The last chunk is short whenever q does
            # not divide the extent -- the same ragged tail the row split has, clamped the
            # same way.
            row0 = Int32(by) * q
            left = project_n - row0
            cnt = left if left < q else q  # noqa: FURB136 -- no DSL builtin min
            # _split_p caps npar, so q * (npar - 1) can exceed the extent and leave the
            # last blocks with nothing. A negative trip count happens to lower to a
            # zero-trip loop, which is not a guarantee worth resting a global load on.
            zero = Int32(0)
            cnt = cnt if cnt > zero else zero  # noqa: FURB136 -- no DSL builtin max
            accs = fold_cols_rolled(
                trait,
                mIns[0],
                unit,
                row0,
                cnt,
                const_expr(self.vec),
                const_expr(self.unroll),
            )
        elif const_expr(self.use_tma):
            accs = (self._fold_tma(mIns[0], tma_atom, bx, tx),)
        elif const_expr(self.tpr == 1):
            # One thread owns the row: no lane to share chunks with, so this is the same fold
            # with the wave/lane arithmetic (and its predicate) removed.
            accs = (
                fold_linear_rolled(
                    trait,
                    mIns[0],
                    unit,
                    const_expr(self.vec),
                    nchunks,
                    const_expr(self.unroll),
                ),
            )
        else:
            acc = fold_row_rolled(
                trait,
                mIns[0],
                unit,
                self.tm,
                lane,
                nchunks,
                nwaves,
                const_expr(self.unroll),
            )
            acc = merge_lanes(trait, acc, const_expr(self.tpr))
            accs = (self._block_merge(acc),)

        # Output indexing comes AFTER the fold: computed before it, these values stay live
        # across the loop and cost registers the fold wants.
        out_base = unit * const_expr(self.nslots)
        if const_expr(self.axis in ("row", "general")):
            # The general axis pins gy to 1, so `by` is always 0 and the col arm below would
            # come out as this anyway -- through a runtime multiply that is always zero. The
            # split shape's partial for (row, batch) sits at row * nbatch + batch, which is
            # exactly how its grid is laid out.
            part_base = (
                Int32(bx)
                if const_expr(
                    self.order == "inner_tree" and self.itree.shape == "split"
                )
                else unit
            )
            part_stride = Int32(1)
        else:
            # COL: (P, C) partials put this chunk's columns in row `by`; (C, P) interleaves
            # them per column, which a block-per-column stage 2 needs (see kernel_coltile).
            part_base = (
                Int32(by) * (nchunks * const_expr(self.nslots)) + out_base
                if const_expr(self.pc)
                else out_base * npar + Int32(by)
            )
            part_stride = Int32(1) if const_expr(self.pc) else npar

        # project OUTSIDE the store branch: binding a dynamic value inside a dynamic `if` is
        # rejected by the DSL, and touching the trait there leaks it into the IR flattener.
        if const_expr(self.final):
            res = tuple(trait.project(a, trait.acc(project_n)) for a in accs)
            if lane == 0 and alive:
                for s in cutlass.range_constexpr(self.nslots):
                    if const_expr(self.nouts == 1):
                        mOuts[0][out_base + Int32(s)] = mOuts[0].element_type(res[s])
                    else:
                        for k in cutlass.range_constexpr(self.nouts):
                            mOuts[k][out_base + Int32(s)] = mOuts[k].element_type(
                                res[s][k]
                            )
        else:
            # RAW per-field partials, for a second stage to combine.
            if lane == 0 and alive:
                for s in cutlass.range_constexpr(self.nslots):
                    for f in cutlass.range_constexpr(trait.nfields):
                        mOuts[f][part_base + Int32(s) * part_stride] = trait.fdtypes[f](
                            accs[s][f]
                        )
