# Shared reduction datapath; drivers own launch/cache policy. Tiles map vec
# elements/load across threads_per_row threads/row. Rolled folds share vector-class kernels;
# fixed DAGs preserve order. Reuse avoids prior 3.7x unwidened and 3x alignment losses.

import math
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64, pipeline
from cutlass.cute.nvgpu import cpasync


WARP = 32
# Wide loads, cp.async, and TMA use the same byte-alignment contract.
TRANSFER_ALIGNMENT = 16

# Static unroll compiles superlinearly past ~1300 operations.
MAX_UNROLL = 512


def vec_size(N: int, itemsize: int) -> int:
    """Elements/load; gcd makes vec divide N and preserves alignment at every chunk."""
    return math.gcd(N, max(1, TRANSFER_ALIGNMENT // itemsize))


def align_bytes(N: int, itemsize: int) -> int:
    """Input alignment declaration; omitting it emits narrow loads and costs 3x."""
    return vec_size(N, itemsize) * itemsize


def _decode_offset(
    linear, divs, strides, npairs, native_div: cutlass.Constexpr = False
):
    # Mixed-radix flat offset with only pair count compiled.
    rem = linear
    if npairs == 0:
        # The operand collapsed to a single element, so every lane maps to offset 0. Without this
        # arm, `strides[npairs - 1]` indexes strides[-1].
        return cutlass.Int64(0)
    if npairs == 1:
        return cutlass.Int64(rem) * strides[0]
    off = cutlass.Int64(0)
    for j in range(npairs - 1):
        if const_expr(native_div):
            q, r = rem // divs[j], rem % divs[j]
        else:
            q, r = divmod(rem, divs[j])
        off = off + cutlass.Int64(r) * strides[j]
        rem = q
    return off + cutlass.Int64(rem) * strides[npairs - 1]


@cute.jit
def _load_decoded(mX, obase, linear, divs, strides, npairs, native_div):
    return mX[obase + _decode_offset(linear, divs, strides, npairs, native_div)]


@cute.jit
def _decoded_index(obase, linear, divs, strides, npairs, native_div, wide):
    idx = obase + _decode_offset(linear, divs, strides, npairs, native_div)
    return idx if const_expr(wide) else Int32(idx)


def _ilog2(n: int) -> int:
    # Both callers pass >= 2; clamping would make n=1 wrong.
    return n.bit_length() - 1


def _off(base, i: int):
    # Keep a static base static (see TileMap.col_base): int + int stays foldable.
    return base + i if isinstance(base, int) else base + Int32(i)


class TileMap:
    """Map row elements to threads and loads; threads_per_row=1 needs no lane merge."""

    def __init__(
        self,
        N: int,
        itemsize: int,
        threads_per_row: int,
        loads: int,
        warp_major: bool = False,
        vec: int | None = None,
        exact: bool | None = None,
    ) -> None:
        # Cross-warp butterflies require a power-of-two warp count or drop partials.
        nw = threads_per_row // WARP
        if threads_per_row != 1 and (threads_per_row % WARP or nw & (nw - 1)):
            raise ValueError(
                f"threads_per_row must be 1 or a power-of-two multiple of {WARP}, got {threads_per_row}"
            )
        unroll = (vec_size(N, itemsize) if vec is None else vec) * loads
        if unroll > MAX_UNROLL:
            raise ValueError(
                f"per-thread unroll {unroll} (vec*loads) exceeds MAX_UNROLL={MAX_UNROLL}; "
                f"compile time scales with it (~1 ms/op, superlinear past ~1300). "
                f"Got N={N} threads_per_row={threads_per_row} loads={loads}."
            )
        self.N = N
        # Fixed DAGs pass an N-independent vec; deriving it from N would change bits.
        self.vec = vec_size(N, itemsize) if vec is None else vec
        self.threads_per_row = threads_per_row
        self.loads = loads
        self.warp_major = warp_major
        self.nw = 1 if threads_per_row == 1 else threads_per_row // WARP
        # Exact full-row tiles need no load predicates; batched tiles are never exact.
        self.exact = (
            (self.vec * self.loads * self.threads_per_row == N)
            if exact is None
            else exact
        )
        # vec must divide N to align every row and chunk; otherwise load by element.
        self.wide_ok = N % self.vec == 0

    @property
    def sig(self) -> tuple[Any, ...]:
        return (
            self.N,
            self.vec,
            self.threads_per_row,
            self.loads,
            self.warp_major,
            self.exact,
            self.wide_ok,
        )

    def align_bytes(self, itemsize: int) -> int:
        """Alignment to declare for THIS tile (element width when the wide load is off)."""
        return self.vec * itemsize if self.wide_ok else itemsize

    def strides(self) -> tuple[int, int, int]:
        """(lane, warp, load) strides; warp_major swaps the warp/load assignment."""
        if self.threads_per_row == 1:
            return (0, 0, self.vec)
        wle = WARP * self.vec  # columns one warp covers in one load
        if self.warp_major:
            return (self.vec, wle * self.loads, wle)
        return (self.vec, wle, wle * self.nw)

    def col_base(self, lane, w, l: int, warp_stride=None):
        """First column of load `l`; preserve static ints so the compiler proves alignment."""
        s_lane, s_w, s_l = self.strides()
        if warp_stride is not None:
            # Caller supplies the per-warp stride; 0 means the warp offset is already folded
            # into base_col.
            s_w = warp_stride
        if self.threads_per_row == 1:
            return l * s_l
        return lane * Int32(s_lane) + w * s_w + Int32(const_expr(l * s_l))


@cute.jit
def _load_reduction_value(trait, mX, off, complex_input: cutlass.Constexpr):
    if const_expr(complex_input):
        base = off * 2
        if const_expr(getattr(trait, "preserve_input_dtype", False)):
            return (mX[base], mX[base + 1])
        return (trait.acc(mX[base]), trait.acc(mX[base + 1]))
    if const_expr(getattr(trait, "preserve_input_dtype", False)):
        return mX[off]
    return trait.acc(mX[off])


@cute.jit
def _store_reduction_value(mOut, index, value, width: cutlass.Constexpr):
    if const_expr(width == 1):
        mOut[index] = mOut.element_type(value)
    else:
        index = index * 2
        mOut[index] = mOut.element_type(value[0])
        mOut[index + 1] = mOut.element_type(value[1])


@cute.jit
def fold_decoded(
    trait,
    mX,
    obase,
    rdivs,
    rstrides,
    npairs: cutlass.Constexpr,
    rb,
    threads_per_block: cutlass.Constexpr,
    tidx,
    in_base,
    chunk_base,
    gidx: cutlass.Constexpr = "r",
    wide_gidx: cutlass.Constexpr = False,
    native_div: cutlass.Constexpr = False,
    complex_input: cutlass.Constexpr = False,
):
    """Grid-stride one output using mixed-radix addressing.

    Runtime decodes cover arbitrary layouts; threads_per_block threads cooperate and gidx selects
    the index reported to traits.
    """
    reduce_fn = trait.reduce
    acc = trait.init()
    n_full = rb // threads_per_block
    base_r = tidx
    for _ in cutlass.range(n_full):
        # Inline offsets to avoid a spurious loop-carried value.
        if const_expr(gidx == "flat"):
            acc = reduce_fn(
                acc,
                _load_reduction_value(
                    trait,
                    mX,
                    obase + _decode_offset(base_r, rdivs, rstrides, npairs, native_div),
                    complex_input,
                ),
                _decoded_index(
                    obase,
                    base_r,
                    rdivs,
                    rstrides,
                    npairs,
                    native_div,
                    wide_gidx,
                ),
                True,
            )
        elif const_expr(gidx == "chunk"):
            acc = reduce_fn(
                acc,
                _load_reduction_value(
                    trait,
                    mX,
                    obase + _decode_offset(base_r, rdivs, rstrides, npairs, native_div),
                    complex_input,
                ),
                chunk_base + base_r,
                True,
            )
        else:
            acc = reduce_fn(
                acc,
                _load_reduction_value(
                    trait,
                    mX,
                    obase + _decode_offset(base_r, rdivs, rstrides, npairs, native_div),
                    complex_input,
                ),
                base_r,
                True,
            )
        base_r = base_r + threads_per_block
    # Invalid lanes read in_base because overhanging chunks may put obase out of range.
    valid = base_r < rb
    off = obase + _decode_offset(base_r, rdivs, rstrides, npairs, native_div)
    off_s = off if valid else in_base
    val = _load_reduction_value(trait, mX, off_s, complex_input)
    if const_expr(gidx == "flat"):
        return reduce_fn(
            acc, val, off_s if const_expr(wide_gidx) else Int32(off_s), valid
        )
    if const_expr(gidx == "chunk"):
        return reduce_fn(acc, val, chunk_base + base_r, valid)
    return reduce_fn(acc, val, base_r, valid)


@cute.jit
def fold_decoded_uniform_tree(
    trait,
    mX,
    obase,
    rstrides,
    threads_per_block: cutlass.Constexpr,
    tidx,
    count: cutlass.Constexpr,
):
    """Fold equal-size Welford leaves without runtime divisions."""
    tree = []
    nloads = const_expr(count // threads_per_block)
    remainder = const_expr(count % threads_per_block)
    stride = rstrides[0]
    for load in cutlass.range_constexpr(nloads):
        r = tidx + Int32(const_expr(load * threads_per_block))
        carry = trait.leaf(
            _load_reduction_value(trait, mX, obase + Int64(r) * stride, False),
            r,
        )
        width = 1
        merges = ((load + 1) & -(load + 1)).bit_length() - 1
        for _ in cutlass.range_constexpr(merges):
            carry = trait.combine_equal(tree.pop(), carry, const_expr(width))
            width *= 2
        tree.append(carry)
    acc = tree[0]
    for i in cutlass.range_constexpr(1, len(tree)):
        acc = trait.combine(acc, tree[i])
    if const_expr(remainder):
        r = Int32(const_expr(nloads * threads_per_block)) + tidx
        valid = tidx < Int32(remainder)
        safe = r if valid else Int32(0)
        leaf = trait.leaf(
            _load_reduction_value(trait, mX, obase + Int64(safe) * stride, False),
            safe,
        )
        ident = trait.init()
        acc = trait.combine(
            acc,
            tuple(
                leaf[f] if valid else ident[f] for f in range(const_expr(trait.nfields))
            ),
        )
    return acc


@cute.jit
def fold_partials_run(
    trait, mIns, obase, rb, threads_per_block: cutlass.Constexpr, tidx, in_base
):
    """Grid-stride one output's partial tuples for stage 2.
    The loop stays dynamic because partial counts reach about 1e5.
    """
    combine_fn = trait.combine
    fdtypes = trait.fdtypes
    nf = const_expr(trait.nfields)
    acc = trait.init()
    n_full = rb // threads_per_block
    r = tidx
    for _ in cutlass.range(n_full):
        rr = obase + cutlass.Int64(r)
        # Bare range unrolls and keeps trait attribute access outside the dynamic loop.
        acc = combine_fn(acc, tuple(fdtypes[f](mIns[f][rr]) for f in range(nf)))
        r = r + threads_per_block
    valid = r < rb
    rr = (obase + cutlass.Int64(r)) if valid else in_base
    part = tuple(fdtypes[f](mIns[f][rr]) for f in range(nf))
    merged = combine_fn(acc, part)
    return tuple((merged[f] if valid else acc[f]) for f in range(nf))


@cute.jit
def merge_lanes(
    trait, acc, threads_per_row: cutlass.Constexpr, asc: cutlass.Constexpr = False
):
    """Merge row lanes; `asc` controls numeric association and index ties. No-op at threads_per_row=1."""
    if const_expr(threads_per_row == 1):
        return acc
    from .traits import warp_reduce

    return warp_reduce(trait, acc, threads_per_row, ascending=asc)


def leaf_op(trait):
    # Trait-owned tuple combiners let index and Welford accumulators share tree folds.
    return trait.combine


def identity(trait):
    return trait.init()


def _linear_reduce(vals, op):
    acc = vals[0]
    for i in range(1, len(vals)):
        acc = op(acc, vals[i])
    return acc


def _inner_tree_reduce(vals, op):
    # Stride-doubling tree: v[i] += v[i + stride] for strides 1, 2, 4, ...
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
    # Match ATen: merge trailing_zeros(load + 1), capped at max_depth, with
    # the existing accumulator on the left.
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
    complex_input: cutlass.Constexpr = False,
):
    """Fold strided or contiguous groups; out-of-row slots add identity to the DAG."""
    op, ident = leaf_op(trait), identity(trait)
    nf = const_expr(trait.nfields)
    tree: list = []
    for i in cutlass.range_constexpr(len(cols)):
        vals = []
        for j in cutlass.range_constexpr(vec):
            # Hoist trait.leaf and select per field: trait access in a dynamic branch
            # leaks the Python object into IR. Unwritten slots are discarded.
            col = _off(cols[i], j)
            x = trait.leaf(
                _load_reduction_value(trait, frag[None, i], j, complex_input),
                col,
            )
            if const_expr(exact):
                vals.append(x)
            else:
                ok = col < hi
                vals.append(tuple(x[f] if ok else ident[f] for f in range(nf)))
        # Linear vec folding changes ATen's association; retained to price its zero benefit.
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
    complex_input: cutlass.Constexpr = False,
):
    """The per-chunk arm: a TileMap's strided groups, one lane butterfly per load."""
    return fold_groups(
        trait,
        frag,
        [tm.col_base(lane, w, l, warp_stride) + base_col for l in range(tm.loads)],
        const_expr(tm.vec),
        Int32(const_expr(tm.N)) if bound is None else bound,
        max_depth,
        const_expr(tm.threads_per_row),
        merge_per_group=True,
        exact=const_expr(tm.exact),
        vec_linear=vec_linear,
        complex_input=complex_input,
    )


# One buffer: smem occupancy matters more than transfer latency.
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
    complex_input: cutlass.Constexpr = False,
):
    """Fold row `r` with a runtime loop, clamping and masking tail waves to avoid DSL branches."""
    reduce_fn = trait.reduce
    acc = trait.init()
    vec = const_expr(tm.vec)
    storage_vec = const_expr(tm.vec * (2 if complex_input else 1))
    threads_per_row = const_expr(tm.threads_per_row)
    gv = cute.flat_divide(mX[Int64(r), None], (storage_vec,))
    frag = cute.make_rmem_tensor(cute.make_layout(storage_vec), mX.element_type)
    for c in cutlass.range(nwaves, unroll=unroll):
        k = c * Int32(threads_per_row) + lane
        ok = k < nchunks
        ks = k if ok else Int32(0)  # clamp so the load is always in range
        cute.autovec_copy(gv[None, ks], frag)
        for i in cutlass.range_constexpr(vec):
            val = _load_reduction_value(trait, frag, i, complex_input)
            acc = reduce_fn(acc, val, ks * Int32(vec) + Int32(i), ok)
    return acc


@cute.jit
def fold_row_aligned(
    trait,
    mX,
    r,
    tm: cutlass.Constexpr,
    lane,
    n,
    unroll: cutlass.Constexpr = _ROLL_UNROLL,
):
    """Fold a ragged row with aligned wide interior loads and scalar edges."""
    reduce_fn = trait.reduce
    acc_dtype = trait.acc
    acc = trait.init()
    vec = const_expr(tm.vec)
    threads_per_row = const_expr(tm.threads_per_row)
    rowv = mX[Int64(r), None]
    aligned = rowv.iterator.align(TRANSFER_ALIGNMENT)
    prefix = Int32((-Int64(r) * Int64(n)) & Int64(vec - 1))
    prefix = n if prefix > n else prefix  # noqa: FURB136
    groups = (n - prefix) // Int32(vec)
    waves = (groups + Int32(threads_per_row - 1)) // Int32(threads_per_row)
    tail = prefix + groups * Int32(vec)
    for i in cutlass.range_constexpr(vec - 1):
        prefix_valid = (lane == Int32(0)) & (Int32(i) < prefix)
        prefix_index = Int32(i) if prefix_valid else Int32(0)
        acc = reduce_fn(
            acc,
            acc_dtype(rowv[prefix_index]),
            prefix_index,
            prefix_valid,
        )
        tail_index = tail + Int32(i)
        tail_valid = (lane == Int32(0)) & (tail_index < n)
        safe_tail = tail_index if tail_valid else Int32(0)
        acc = reduce_fn(
            acc,
            acc_dtype(rowv[safe_tail]),
            safe_tail,
            tail_valid,
        )
    frag = cute.make_rmem_tensor(cute.make_layout(vec), mX.element_type)
    for c in cutlass.range(waves, unroll=unroll):
        group = c * Int32(threads_per_row) + lane
        valid = group < groups
        safe_group = group if valid else Int32(0)
        base = prefix + safe_group * Int32(vec)
        src = cute.make_tensor(
            aligned + safe_group * Int32(vec),
            cute.make_layout(vec),
        )
        cute.autovec_copy(src, frag)
        for i in cutlass.range_constexpr(vec):
            acc = reduce_fn(
                acc,
                acc_dtype(frag[i]),
                base + Int32(i),
                valid,
            )
    return acc


@cute.jit
def fold_row_prefetched4(
    trait,
    mX,
    r,
    tm: cutlass.Constexpr,
    lane,
    nchunks,
    nwaves,
    complex_input: cutlass.Constexpr = False,
):
    """Fold four independent loads at a time to hide global-memory latency."""
    reduce_fn = trait.reduce
    acc = trait.init()
    threads_per_row = const_expr(tm.threads_per_row)
    row = mX[Int64(r), None]
    nbatches = nwaves // Int32(4)
    for batch in cutlass.range(nbatches):
        k0 = (batch * Int32(4)) * Int32(threads_per_row) + lane
        k1 = (batch * Int32(4) + Int32(1)) * Int32(threads_per_row) + lane
        k2 = (batch * Int32(4) + Int32(2)) * Int32(threads_per_row) + lane
        k3 = (batch * Int32(4) + Int32(3)) * Int32(threads_per_row) + lane
        v0 = k0 < nchunks
        v1 = k1 < nchunks
        v2 = k2 < nchunks
        v3 = k3 < nchunks
        s0 = k0 if v0 else Int32(0)
        s1 = k1 if v1 else Int32(0)
        s2 = k2 if v2 else Int32(0)
        s3 = k3 if v3 else Int32(0)
        x0 = _load_reduction_value(trait, row, s0, complex_input)
        x1 = _load_reduction_value(trait, row, s1, complex_input)
        x2 = _load_reduction_value(trait, row, s2, complex_input)
        x3 = _load_reduction_value(trait, row, s3, complex_input)
        acc = reduce_fn(acc, x0, s0, v0)
        acc = reduce_fn(acc, x1, s1, v1)
        acc = reduce_fn(acc, x2, s2, v2)
        acc = reduce_fn(acc, x3, s3, v3)
    first_tail = nbatches * Int32(4)
    for c in cutlass.range(nwaves - first_tail):
        k = (first_tail + Int32(c)) * Int32(threads_per_row) + lane
        valid = k < nchunks
        ks = k if valid else Int32(0)
        acc = reduce_fn(
            acc,
            _load_reduction_value(trait, row, ks, complex_input),
            ks,
            valid,
        )
    return acc


@cute.jit
def fold_linear_rolled(
    trait,
    mX: cute.Tensor,
    r: Int32,
    vec: cutlass.Constexpr,
    nchunks: Int32,
    unroll: cutlass.Constexpr = _ROLL_UNROLL,
    complex_input: cutlass.Constexpr = False,
):
    """Fold row `r` with a runtime chunk loop. Returns an acc tuple. threads_per_row == 1 only."""
    reduce_fn = trait.reduce
    acc = trait.init()
    storage_vec = const_expr(vec * (2 if complex_input else 1))
    gv = cute.flat_divide(mX[Int64(r), None], (storage_vec,))
    frag = cute.make_rmem_tensor(cute.make_layout(storage_vec), mX.element_type)
    for c in cutlass.range(nchunks, unroll=unroll):
        cute.autovec_copy(gv[None, c], frag)
        for i in cutlass.range_constexpr(vec):
            val = _load_reduction_value(trait, frag, i, complex_input)
            acc = reduce_fn(acc, val, c * Int32(vec) + Int32(i), True)
    return acc


@cute.jit
def _wide(rowv, base, vec: cutlass.Constexpr, frag_l):
    # Load up to 128 bits; pointer offset permits a dynamic base.
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
    complex_input: cutlass.Constexpr = False,
):
    """Fill this thread's (vec, loads) fragment.

    Use one wide load when possible; ragged groups use predicated scalar reads.
    """
    hi = Int32(const_expr(tm.N)) if bound is None else bound
    # Exact covers a full row from column 0; bounds and shifted bases require predication.
    whole_row = const_expr(
        tm.exact and bound is None and isinstance(base_col, int) and base_col == 0
    )
    rowv = mX[Int64(r), None]
    storage_scale = const_expr(2 if complex_input else 1)
    storage_vec = const_expr(tm.vec * storage_scale)
    for l in cutlass.range_constexpr(tm.loads):
        # Plain addition preserves a static base and promotes a dynamic one.
        base = tm.col_base(lane, w, l, warp_stride) + base_col
        storage_base = base * storage_scale
        if const_expr(whole_row and tm.wide_ok):
            _wide(rowv, storage_base, storage_vec, frag[None, l])
        elif const_expr(not tm.wide_ok):
            for i in cutlass.range_constexpr(tm.vec):
                # Inline because the DSL rejects binding a dynamic value in a dynamic branch.
                if _off(base, i) < hi:
                    if const_expr(complex_input):
                        frag[2 * i, l] = rowv[_off(storage_base, 2 * i)]
                        frag[2 * i + 1, l] = rowv[_off(storage_base, 2 * i + 1)]
                    else:
                        frag[i, l] = rowv[_off(base, i)]
        else:
            if _off(base, tm.vec) <= hi:
                _wide(rowv, storage_base, storage_vec, frag[None, l])
            else:
                for i in cutlass.range_constexpr(tm.vec):
                    if _off(base, i) < hi:
                        if const_expr(complex_input):
                            frag[2 * i, l] = rowv[_off(storage_base, 2 * i)]
                            frag[2 * i + 1, l] = rowv[_off(storage_base, 2 * i + 1)]
                        else:
                            frag[i, l] = rowv[_off(base, i)]


@cute.jit
def load_decoded(
    mX,
    obase,
    tm: cutlass.Constexpr,
    lane,
    w,
    frag,
    rdivs,
    rstrides,
    npairs: cutlass.Constexpr,
    in_base,
    base_col=0,
    bound=None,
    warp_stride=None,
    complex_input: cutlass.Constexpr = False,
):
    """Gather logical row elements through a mixed-radix storage mapping."""
    hi = Int32(const_expr(tm.N)) if bound is None else bound
    for l in cutlass.range_constexpr(tm.loads):
        base = tm.col_base(lane, w, l, warp_stride) + base_col
        for i in cutlass.range_constexpr(tm.vec):
            col = _off(base, i)
            valid = col < hi
            logical = col if valid else Int32(0)
            off = obase + _decode_offset(logical, rdivs, rstrides, npairs)
            src = off if valid else in_base
            if const_expr(complex_input):
                frag[2 * i, l] = mX[src * 2]
                frag[2 * i + 1, l] = mX[src * 2 + 1]
            else:
                frag[i, l] = mX[src]


def make_fragment(mX, tm, complex_input: cutlass.Constexpr = False) -> cute.Tensor:
    storage_vec = tm.vec * (2 if complex_input else 1)
    return cute.make_rmem_tensor(
        cute.make_layout((storage_vec, tm.loads)), mX.element_type
    )


@cute.jit
def fold_row_static(
    trait, mX, row, tm: cutlass.Constexpr, lane, accumulators: cutlass.Constexpr
):
    reduce_fn, combine_fn = trait.reduce, trait.combine
    frag = make_fragment(mX, tm)
    warp_lane, warp = lane % Int32(WARP), lane // Int32(WARP)
    load(mX, row, tm, warp_lane, warp, frag)
    accs = [trait.init() for _ in range(accumulators)]
    for l in cutlass.range_constexpr(tm.loads):
        base = tm.col_base(warp_lane, warp, l)
        for i in cutlass.range_constexpr(tm.vec):
            slot = const_expr((l * tm.vec + i) % accumulators)
            accs[slot] = reduce_fn(
                accs[slot], trait.acc(frag[i, l]), base + Int32(i), True
            )
    acc = accs[0]
    for i in cutlass.range_constexpr(1, accumulators):
        acc = combine_fn(acc, accs[i])
    return acc


def smem_box_layout(N: int, threads: int) -> cute.Layout:
    """Plain row-major smem for a (threads, N) TMA box.

    TMA's GEMM swizzles do not fix whole-row bank conflicts; fold_smem_rotated does.
    """
    return cute.make_ordered_layout((threads, N), order=(1, 0))


@cute.jit
def fold_smem_rotated(
    trait,
    sX,
    rb,
    N: cutlass.Constexpr,
    complex_input: cutlass.Constexpr = False,
):
    """Fold one staged row per thread, rotating reads by row to avoid 32-way bank conflicts.

    Pass logical columns to the trait; power-of-two N makes rotation a mask.
    """
    acc = trait.init()
    mask = const_expr(N - 1)
    row = sX[rb, None]
    for c in cutlass.range_constexpr(N):
        col = (Int32(c) + rb) & Int32(mask)
        val = _load_reduction_value(trait, row, col, complex_input)
        acc = trait.reduce(acc, val, col, True)
    return acc


@cute.jit
def fold_cols_rolled(
    trait,
    mX,
    col,
    storage_row0,
    index_row0,
    nrows,
    vec: cutlass.Constexpr,
    unroll: cutlass.Constexpr = _ROLL_UNROLL,
    complex_input: cutlass.Constexpr = False,
):
    """Fold rows into vec kept-axis accumulators without lane merging."""
    reduce_fn = trait.reduce
    accs = tuple(trait.init() for _ in range(vec))
    storage_vec = const_expr(vec * (2 if complex_input else 1))
    frag = cute.make_rmem_tensor(cute.make_layout(storage_vec), mX.element_type)
    for r in cutlass.range(nrows, unroll=unroll):
        rr = storage_row0 + Int32(r)
        cute.autovec_copy(
            cute.flat_divide(mX[Int64(rr), None], (storage_vec,))[None, col], frag
        )
        # The DSL does not preprocess comprehensions; plain range still unrolls constexpr vec.
        accs = tuple(
            reduce_fn(
                accs[i],
                _load_reduction_value(trait, frag, i, complex_input),
                index_row0 + Int32(r),
                True,
            )
            for i in range(vec)
        )
    return accs


@cute.jit
def _fold_welford_equal_group(
    leaf_fn,
    combine_equal_fn,
    mX,
    col,
    row0,
    idx0,
    group: cutlass.Constexpr,
):
    tree = []
    for i in cutlass.range_constexpr(group):
        row = mX[Int64(row0 + Int32(i)), None]
        tree.append(leaf_fn(row[col], idx0 + Int32(i)))
    width = 1
    for _ in cutlass.range_constexpr(group.bit_length() - 1):
        tree = [
            combine_equal_fn(tree[2 * i], tree[2 * i + 1], const_expr(width))
            for i in range(len(tree) // 2)
        ]
        width *= 2
    return tree[0]


@cute.jit
def fold_col_welford_grouped(
    trait,
    mX,
    col,
    storage_row0,
    index_row0,
    nrows,
    group: cutlass.Constexpr,
):
    """Fold fixed-size row groups as equal-count Welford trees."""
    reduce_fn = trait.reduce
    leaf_fn = trait.leaf
    combine_fn = trait.combine
    combine_equal_fn = trait.combine_equal
    acc = trait.init()
    tail = nrows % Int32(group)
    frag = cute.make_rmem_tensor(cute.make_layout(1), mX.element_type)
    for i in cutlass.range_constexpr(group - 1):
        valid = Int32(i) < tail
        rr = storage_row0 + Int32(i) if valid else Int32(0)
        cute.autovec_copy(cute.flat_divide(mX[Int64(rr), None], (1,))[None, col], frag)
        acc = reduce_fn(
            acc,
            trait.acc(frag[0]),
            index_row0 + Int32(i),
            valid,
        )
    ngroups = (nrows - tail) // Int32(group)
    for g in cutlass.range(ngroups):
        row0 = storage_row0 + tail + g * Int32(group)
        idx0 = index_row0 + tail + g * Int32(group)
        root = _fold_welford_equal_group(
            leaf_fn,
            combine_equal_fn,
            mX,
            col,
            row0,
            idx0,
            const_expr(group),
        )
        acc = combine_fn(acc, root)
    return (acc,)


class TileReduce:
    """Reduction kernel parameterized by reduced axis.

    Row uses threads_per_row threads per contiguous row and merges lanes. Column gives each thread
    vec adjacent outputs and splits strided rows on grid y. Only folding is axis-specific;
    dead-thread handling, projection, and nslots-by-nouts stores are shared.
    """

    def __init__(
        self,
        trait,
        dtype,
        axis,
        N,
        threads_per_row=1,
        threads_per_block=128,
        nouts=1,
        final=True,
        unroll=_ROLL_UNROLL,
        load_ahead=1,
        vec: int | None = None,
        use_tma=False,
        combine=False,
        batched_col=False,
        pc=True,
        npairs_red=0,
        npairs_kept=0,
        wide_count=False,
        wide_output=False,
        wide_gidx=False,
        wide_red=False,
        wide_kept=False,
        gidx_from="r",
        flat_tail=False,
        ragged_chunk=False,
        order="linear",
        # Duck-typed because its driver-owned type would invert the dependency.
        itree: Any = None,
        row_accumulators: int = 0,
        ragged_vector=False,
        general_tree_count: int = 0,
        col_group: int = 1,
    ) -> None:
        if axis not in ("row", "col", "general"):
            raise ValueError(f"axis must be 'row', 'col' or 'general', got {axis!r}")
        if order not in ("linear", "inner_tree"):
            raise ValueError(f"order must be 'linear' or 'inner_tree', got {order!r}")
        if row_accumulators not in (0, 1, 2, 4, 8):
            raise ValueError("row_accumulators must be 0, 1, 2, 4, or 8")
        if general_tree_count and (
            axis != "general"
            or order != "linear"
            or combine
            or npairs_red != 1
            or wide_gidx
            or wide_red
            or gidx_from != "r"
            or getattr(trait, "complex_input", False)
            or not hasattr(trait, "combine_equal")
            or general_tree_count < threads_per_block
        ):
            raise ValueError("general tree needs a nonempty Welford thread fold")
        if (
            col_group not in (1, 4, 8, 16)
            or col_group > 1
            and (
                axis != "col"
                or combine
                or vec != 1
                or getattr(trait, "complex_input", False)
                or not hasattr(trait, "combine_equal")
            )
        ):
            raise ValueError("column groups need scalar real Welford columns")
        if row_accumulators and (
            axis != "row"
            or order != "linear"
            or use_tma
            or combine
            or load_ahead != 1
            or threads_per_row < WARP
            or getattr(trait, "complex_input", False)
            or N <= 0
            or N % (threads_per_row * vec_size(N, dtype.width // 8))
            or N // threads_per_row < row_accumulators
        ):
            raise ValueError(
                "static row accumulators require complete real-valued linear row tiles"
            )
        if ragged_vector and (
            axis != "row"
            or order != "linear"
            or vec is None
            or vec <= 1
            or vec & (vec - 1)
            or use_tma
            or combine
            or load_ahead != 1
            or row_accumulators
            or getattr(trait, "complex_input", False)
        ):
            raise ValueError("ragged vectors require a real linear row reduction")
        if order == "inner_tree":
            if axis not in ("row", "general") or itree is None:
                raise ValueError(
                    "the inner-tree order needs a row or general axis and a plan"
                )
            # The plan maps wpr chunks to wpr // kchunk warps and rows_per_block rows.
            threads_per_row = WARP * (itree.wpr // itree.kchunk) if itree.wpr else 1
            if itree.stage_e:
                threads_per_row = WARP
            threads_per_block = threads_per_row * itree.rows_per_block
            if itree.shape == "combine" and itree.combine_tile:
                # The async transpose needs every lane even for a short row group.
                threads_per_block = max(WARP, threads_per_block)
        if axis == "general" and order == "linear":
            # Every block thread folds one output, equivalent to row threads_per_row=threads_per_block.
            threads_per_row = threads_per_block
            nwg = threads_per_block // WARP
            if threads_per_block % WARP or nwg & (nwg - 1):
                # Non-power-of-two warp counts omit partials in _block_merge, measuring
                # 12-62% low; caller-set block requires validation here.
                raise ValueError(
                    f"a general-axis block must be a power-of-two warp count, got {threads_per_block=}"
                )
        nwpr = threads_per_row // WARP
        if (
            axis == "row"
            and threads_per_row != 1
            and (
                threads_per_row % WARP
                or threads_per_row > threads_per_block
                or threads_per_block % threads_per_row
                or nwpr & (nwpr - 1)
            )
        ):
            raise ValueError(
                f"threads_per_row must be 1 or a power-of-two multiple of {WARP} dividing threads_per_block: {threads_per_row=} {threads_per_block=}"
            )
        if use_tma and (axis != "row" or threads_per_row != 1):
            raise ValueError(
                f"TMA stages whole rows: needs row at threads_per_row 1, {axis=} {threads_per_row=}"
            )
        if use_tma and N & (N - 1):
            # The rotation mask duplicates and skips columns unless N is a power of two;
            # caller-set use_tma can bypass tma_ok.
            raise ValueError(f"TMA staging needs a power-of-two row length, got {N=}")
        self.trait = trait
        self.complex_input = getattr(trait, "complex_input", False)
        self.output_widths = (
            tuple(getattr(trait, "output_widths", (1,) * nouts))
            if final
            else (1,) * nouts
        )
        if len(self.output_widths) != nouts or any(
            width not in (1, 2) for width in self.output_widths
        ):
            raise ValueError(
                f"output widths must contain one 1 or 2 per output, got {self.output_widths}"
            )
        self.dtype = dtype
        self.axis = axis
        self.N = N
        self.threads_per_row = threads_per_row
        self.threads_per_block = threads_per_block
        self.nouts = nouts
        self.final = final
        self.unroll = unroll
        self.load_ahead = load_ahead
        self.row_accumulators = row_accumulators
        self.ragged_vector = ragged_vector
        self.general_tree_count = general_tree_count
        self.col_group = col_group
        self.use_tma = use_tma
        self.combine = combine
        self.batched_col = batched_col
        self.pc = pc  # partial layout: (P, C) when True, else (C, P)
        # Compile-time fold_decoded policy.
        self.npairs_red = npairs_red
        self.npairs_kept = npairs_kept
        self.wide_count = wide_count
        self.wide_output = wide_output
        self.wide_gidx = wide_gidx
        self.wide_red = wide_red
        self.wide_kept = wide_kept
        self.gidx_from = gidx_from
        self.flat_tail = flat_tail
        self.ragged_chunk = ragged_chunk
        self.order = order
        self.itree = itree
        self.storage_width = N * (2 if self.complex_input else 1)
        itemsize = (
            dtype.width // 8 * (2 if self.complex_input else 1)
            if dtype is not None
            else 0
        )
        # Runtime row folds map one load; TMA declares static staged depth. Column and
        # general axes choose no tile.
        self.tm = (
            TileMap(
                N,
                itemsize,
                threads_per_row,
                N // vec_size(N, itemsize)
                if use_tma
                else N // (threads_per_row * vec_size(N, itemsize))
                if row_accumulators
                else 1,
                vec=vec if ragged_vector else None,
                exact=False if ragged_vector else None,
            )
            if axis == "row" and order == "linear"
            else None
        )
        if order == "inner_tree":
            self.vec = itree.vec
        elif axis == "row":
            self.vec = self.tilemap.vec if order == "linear" else itree.vec
        elif axis == "general":
            # Arbitrary strides offer no vector width.
            self.vec = 1
        elif vec is None:
            # Column vec cannot be inferred from load width.
            raise ValueError("the col axis needs an explicit vec")
        else:
            self.vec = vec
        if self.load_ahead != 1 and (
            axis != "row" or self.vec != 1 or self.load_ahead != 4
        ):
            raise ValueError(
                f"load_ahead supports four independent scalar row loads, got "
                f"{axis=} {self.vec=} {self.load_ahead=}"
            )
        # Columns keep vec slots; merged row and general axes keep one.
        self.nslots = self.vec if axis == "col" else 1
        self.rows_per_block = (
            itree.rows_per_block
            if order == "inner_tree" and itree.shape == "combine" and itree.combine_tile
            else threads_per_block // threads_per_row
        )
        self.warps_per_row = (
            threads_per_row // WARP
        )  # 0 at threads_per_row == 1: nothing to merge
        # TMA sees the underlying real descriptor, including both storage scalars
        # for each logical complex element.
        self.tiler = (threads_per_block, self.storage_width)

    @property
    def tilemap(self) -> TileMap:
        # Only linear row folds own one tile; inner-tree plans own one per batch.
        if self.tm is None:
            raise AssertionError(f"no tile on the {self.axis} axis")
        return self.tm

    @property
    def cache_sig(self) -> tuple[Any, ...]:
        # Static row folds and TMA key N; runtime paths share a vector class.
        return (
            self.axis,
            self.vec,
            self.threads_per_row,
            self.threads_per_block,
            self.nouts,
            self.final,
            self.unroll,
            self.load_ahead,
            self.use_tma,
            self.combine,
            self.batched_col,
            self.pc,
            self.trait.nfields,
            self.complex_input,
            self.output_widths,
            getattr(self.trait, "canonical_bool", False),
            self.N
            if self.use_tma or self.row_accumulators or self.ragged_vector
            else 0,
            self.npairs_red,
            self.npairs_kept,
            self.wide_count,
            self.wide_output,
            self.wide_gidx,
            self.wide_red,
            self.wide_kept,
            self.gidx_from,
            self.flat_tail,
            self.ragged_chunk,
            self.ragged_vector,
            self.general_tree_count,
            self.col_group,
            self.order,
            # Fixed DAGs key on N, requiring one kernel per shape instead of per vec class.
            self.itree.sig if self.itree is not None else None,
        ) + ((self.row_accumulators,) if self.row_accumulators else ())

    @cute.jit
    def _fold_tma(self, mX, tma_atom, bx, tx):
        # Transfer whole rows to smem once, then fold with rotated reads.
        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            self.dtype,
            smem_box_layout(self.storage_width, self.threads_per_block),
            byte_alignment=TRANSFER_ALIGNMENT,
        )
        mbar = smem.allocate_array(cutlass.Int64, num_elems=2)
        gX = cute.local_tile(mX, self.tiler, (cutlass.Int64(bx), 0))
        pipe = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, const_expr(self.threads_per_block)
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
        # All of warp 0 must enter: the pipeline signals on lane 0 and synchronizes the warp.
        # One producer thread deadlocks. Rows past M are zero-filled and discarded.
        if cute.arch.warp_idx() == 0:
            pipe.producer_acquire(pstate)
            cute.copy(
                tma_atom, tSgX, tSsX, tma_bar_ptr=pipe.producer_get_barrier(pstate)
            )
            pipe.producer_commit(pstate)
        pipe.consumer_wait(cstate)
        acc = fold_smem_rotated(
            self.trait,
            sX,
            tx,
            const_expr(self.N),
            const_expr(self.complex_input),
        )
        pipe.consumer_release(cstate)
        return acc

    @cute.jit
    def _block_merge(self, acc):
        # Merge one output's warps through smem; one warp is a no-op.
        if const_expr(self.warps_per_row <= 1):
            return acc
        from .traits import block_reduce

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
    def _fold_itree(
        self,
        mX,
        r,
        lane_w,
        warp_id,
        row_in_block,
        batch_idx=None,
        obase=None,
        rdivs=None,
        rstrides=None,
        in_base=None,
    ):
        """Fold static batch fragments by streaming tree, then batches linearly.

        Per-warp results meet in smem via ascending butterfly. Split selects one runtime
        batch per block; all other batch structure is compiled into the DAG.
        """
        trait = self.trait
        it = self.itree
        nf = const_expr(trait.nfields)
        op, ident = leaf_op(trait), identity(trait)
        warp_writes = []  # empty when one warp covers the row: nothing to stage
        if const_expr(it.wpr // max(it.kchunk, 1) > 1):
            # Stage each field separately because their dtypes may differ.
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
        # Only looped seeds with identity, matching upstream signed-zero behavior.
        final = None
        kc = const_expr(it.kchunk)  # ADJACENT CHUNKS this thread group folds
        groups = const_expr(it.wpr // it.kchunk) if it.wpr else 0
        for b in cutlass.range_constexpr(len(it.tms)):
            tm = it.tms[b]
            # Load every chunk first; folding each immediately leaves one load in flight.
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
                    # Bound each chunk so a short one cannot consume the next; this matches
                    # ATen zeroing whole loads beyond its share.
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
                frag = make_fragment(mX, tm, self.complex_input)
                if const_expr(self.axis == "general"):
                    load_decoded(
                        mX,
                        obase,
                        tm,
                        lane_w,
                        cid,
                        frag,
                        rdivs,
                        rstrides,
                        const_expr(self.npairs_red),
                        in_base,
                        base,
                        bound,
                        wstride,
                        const_expr(self.complex_input),
                    )
                else:
                    load(
                        mX,
                        r,
                        tm,
                        lane_w,
                        cid,
                        frag,
                        base,
                        bound,
                        wstride,
                        const_expr(self.complex_input),
                    )
                frags.append(frag)
                bases.append(base)
                bounds.append(bound)
                wstrides.append(wstride)
            # Fold each chunk, then its adjacent chunks as the cross-chunk tree would.
            # This replaces shuffles without changing the DAG or load coalescing.
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
                    const_expr(self.complex_input),
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
                # Clamp inactive lanes because the buffer may hold fewer than 32 entries.
                live = lane_w < Int32(groups)
                idx = slot + (lane_w if live else Int32(0))
                got = tuple(warp_writes[f][idx] for f in range(nf))
                merged = tuple(got[f] if live else ident[f] for f in range(nf))
                warp_acc = merge_lanes(
                    trait, merged, const_expr(tm.threads_per_row), asc=True
                )
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
    def _fold_itree_smem(self, mX, r, lane, row_in_block, batch_idx=None):
        """Stage coalesced tiles in smem for contiguous per-lane folds and one butterfly each.

        One buffer keeps smem fixed; refill follows folding to avoid a large-M write-after-read
        race. Padding each lane by vec separates bank groups. The DAG is unchanged.
        """
        trait = self.trait
        it = self.itree
        # fold_groups owns leaf/mask/tree work; this body carries tiles and seeds identity.
        op, ident = leaf_op(trait), identity(trait)
        Es = const_expr(it.stage_e)  # columns per lane per tile
        vec = const_expr(it.vec)
        span = const_expr(it.batches[0][2] * it.wpr * WARP * it.vec)
        wle = const_expr(WARP * vec)
        tile_cols = const_expr(Es * WARP)
        ntiles = const_expr(span // tile_cols)
        pitch = const_expr(Es + vec)
        stride = const_expr(pitch * WARP)  # one row buffer
        depth = const_expr(min(_ITREE_STAGE_DEPTH, ntiles))
        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            self.dtype,
            cute.make_layout(const_expr(stride * depth * it.rows_per_block)),
            byte_alignment=TRANSFER_ALIGNMENT,
        )
        base = row_in_block * Int32(const_expr(stride * depth))
        rowv = mX[Int64(r), None]
        hi = Int32(const_expr(it.batches[0][1]))  # this batch's real column count
        gv = cute.flat_divide(rowv, (vec,))
        group_base = (
            batch_idx * Int32(const_expr(it.split[1] // it.vec))
            if const_expr(it.shape == "split")
            else Int32(0)
        )
        batch_base = group_base * Int32(vec)
        sv = cute.flat_divide(sX, (vec,))
        g2s = cute.make_copy_atom(
            cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=const_expr(vec * mX.element_type.width),
        )
        runfrag = cute.make_rmem_tensor(
            cute.make_layout((vec, const_expr(Es // vec))), mX.element_type
        )

        # cp.async bypasses registers and needs aligned source and destination; flat_divide
        # preserves the wrap's static 16-byte alignment.
        for pre in cutlass.range_constexpr(depth):
            for i in cutlass.range_constexpr(tile_cols // wle):
                off = const_expr(pre * tile_cols + i * wle)
                col = Int32(const_expr(off)) + lane * Int32(vec)
                k = Int32(const_expr(off // vec)) + lane
                ks = group_base + (
                    k if col + Int32(vec) <= hi else Int32(0)
                )  # clamp a short batch
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
            # Wait for this tile.
            cute.arch.cp_async_wait_group(const_expr(min(depth - 1, ntiles - 1 - t)))
            cute.arch.sync_warp()

            # Fold each lane's contiguous run, then one butterfly.
            run = base + Int32(const_expr((t % depth) * stride)) + lane * Int32(pitch)
            # Index-bearing traits need row-relative indices, including the split batch.
            col0 = batch_base + Int32(const_expr(t * tile_cols)) + lane * Int32(Es)
            # Materialize the run for fold_groups, merging lanes only at the end.
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
                    batch_base + hi,
                    const_expr(_ilog2(Es // vec)),
                    WARP,
                    merge_per_group=False,
                    complex_input=False,
                ),
                t,
                const_expr(_ilog2(max(ntiles, 2))),
                op,
            )
            if const_expr(t + depth < ntiles):
                # Refill only after folding; earlier issue races the read once M is large.
                cute.arch.sync_warp()
                nxt = const_expr(t + depth)
                for i in cutlass.range_constexpr(tile_cols // wle):
                    off = const_expr(nxt * tile_cols + i * wle)
                    col = Int32(const_expr(off)) + lane * Int32(vec)
                    k = Int32(const_expr(off // vec)) + lane
                    ks = group_base + (k if col + Int32(vec) <= hi else Int32(0))
                    local = Int32(const_expr(i * wle)) + lane * Int32(vec)
                    dst = (
                        base
                        + Int32(const_expr((nxt % depth) * stride))
                        + (local // Int32(Es)) * Int32(pitch)
                        + local % Int32(Es)
                    )
                    cute.copy(g2s, gv[None, ks], sv[None, dst // Int32(vec)])
                cute.arch.cp_async_commit_group()
        # Looped seeds identity; split writes the batch tree directly.
        return tree[0] if const_expr(it.shape == "split") else op(ident, tree[0])

    @cute.jit
    def _fold_itree_rowstage(self, mX, tx, bx, r_in_block, lane_w, warp_id, batch_idx):
        """Coalesce a block's adjacent rows through shared memory before the same fold."""
        it = self.itree
        N = const_expr(self.N)
        rpb = const_expr(it.rows_per_block)
        vec = const_expr(it.vec)
        cols = const_expr(N // vec)  # vec groups per row
        ngroups = const_expr(rpb * cols)
        smem = cutlass.utils.SmemAllocator()
        # ROW-MAJOR, spelled out: cute's default for a shape tuple is column-major, which staged the
        # tile transposed and returned plausible wrong numbers.
        sX = smem.allocate_tensor(
            self.dtype,
            cute.make_layout((rpb, N), stride=(N, 1)),
            byte_alignment=TRANSFER_ALIGNMENT,
        )
        # The tile as ONE flat run of vec groups: consecutive counters are consecutive addresses, so
        # a warp covers 32*vec contiguous elements even though a row is narrower than that.
        row0 = Int32(bx) * Int32(const_expr(rpb))
        sv = cute.flat_divide(
            cute.make_tensor(sX.iterator, cute.make_layout(const_expr(rpb * N))), (vec,)
        )
        g2s = cute.make_copy_atom(
            cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=const_expr(vec * mX.element_type.width),
        )
        # A short LAST block's surplus groups clamp onto group 0 rather than branching; the rows
        # they would have filled belong to threads whose store is guarded anyway.
        nlive = (Int32(mX.shape[0]) - row0) * Int32(const_expr(cols))
        for i in cutlass.range_constexpr(
            -(-ngroups // const_expr(self.threads_per_block))
        ):
            g = Int32(const_expr(i * self.threads_per_block)) + tx
            ok = (g < nlive) & (g < Int32(const_expr(ngroups)))
            gs = g if ok else Int32(0)
            # Per ROW, not off a flat pointer: cp.async needs a statically transfer-aligned source,
            # and a raw `iterator + offset` carries no alignment while a flat_divide of a row does.
            gvr = cute.flat_divide(
                mX[Int64(row0 + gs // Int32(const_expr(cols))), None], (vec,)
            )
            cute.copy(g2s, gvr[None, gs % Int32(const_expr(cols))], sv[None, gs])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        return self._fold_itree(sX, r_in_block, lane_w, warp_id, r_in_block, batch_idx)

    @cute.jit
    def _fold_itree_combine_async(self, mIns, row, lane, blk):
        """Coalesce split partials through shared memory without changing fold order."""
        combine_fn, fdtypes = self.trait.combine, self.trait.fdtypes
        nf = const_expr(self.trait.nfields)
        nbatch = const_expr(self.itree.split[0])
        grp = const_expr(self.itree.combine_grp)
        rpb = const_expr(self.itree.rows_per_block)
        tile_n = const_expr(self.itree.combine_tile)  # partials per row per tile
        unroll = const_expr(self.itree.combine_unroll)
        mult = const_expr(tile_n // (WARP * grp))  # cp.async per lane per row
        ntiles = const_expr(nbatch // tile_n)
        # Pad each row's run by `grp` so a lane's run stays transfer-aligned while the runs start in
        # distinct bank groups -- the layout _fold_itree_smem uses.
        pitch = const_expr(tile_n + grp)
        smem = cutlass.utils.SmemAllocator()
        sbuf = [
            smem.allocate_tensor(
                fdtypes[f],
                cute.make_layout(const_expr(rpb * pitch)),
                byte_alignment=TRANSFER_ALIGNMENT,
            )
            for f in range(nf)
        ]
        g2s = [
            cute.make_copy_atom(
                cpasync.CopyG2SOp(),
                fdtypes[f],
                num_bits_per_copy=const_expr(grp * fdtypes[f].width),
            )
            for f in range(nf)
        ]
        gv = [cute.flat_divide(mIns[f], (grp,)) for f in range(nf)]
        sv = [cute.flat_divide(sbuf[f], (grp,)) for f in range(nf)]
        nrows = mIns[0].shape[0] // Int32(const_expr(nbatch))
        row0 = Int32(blk) * Int32(const_expr(rpb))
        live_lane = lane if lane < Int32(const_expr(rpb)) else Int32(0)
        run = live_lane * Int32(const_expr(pitch))
        take = lambda j: tuple(  # noqa: E731
            fdtypes[f](sbuf[f][run + Int32(j)]) for f in range(nf)
        )

        # A tile of every row in the block, one coalesced cp.async per (row, lane). INLINED at both
        # call sites: the DSL rejects a closure inside dynamic control flow, whatever it captures.
        for i in cutlass.range_constexpr(rpb):
            r = row0 + Int32(const_expr(i))
            if r < nrows:
                base_g = r * Int32(const_expr(nbatch))
                for u in cutlass.range_constexpr(mult):
                    src = (
                        base_g // Int32(const_expr(grp))
                        + lane
                        + Int32(const_expr(u * WARP))
                    )
                    dst = Int32(const_expr(i * pitch // grp + u * WARP)) + lane
                    for f in cutlass.range_constexpr(nf):
                        cute.copy(g2s[f], gv[f][None, src], sv[f][None, dst])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        # Tile 0 SEEDS the chain from partial 0 -- never from the identity, since `0.0 + -0.0` is
        # `+0.0` and this fold's first value can be a negative zero.
        acc = take(0)
        if const_expr(unroll):
            for j in cutlass.range(1, tile_n, unroll=unroll):
                acc = combine_fn(acc, take(j))
        else:
            for j in cutlass.range_constexpr(tile_n - 1):
                acc = combine_fn(acc, take(const_expr(j + 1)))
        for t in cutlass.range(1, ntiles):
            cute.arch.barrier()  # the refill overwrites what the previous fold just read
            for i in cutlass.range_constexpr(rpb):
                r = row0 + Int32(const_expr(i))
                if r < nrows:
                    base_g = r * Int32(const_expr(nbatch)) + t * Int32(
                        const_expr(tile_n)
                    )
                    for u in cutlass.range_constexpr(mult):
                        src = (
                            base_g // Int32(const_expr(grp))
                            + lane
                            + Int32(const_expr(u * WARP))
                        )
                        dst = Int32(const_expr(i * pitch // grp + u * WARP)) + lane
                        for f in cutlass.range_constexpr(nf):
                            cute.copy(g2s[f], gv[f][None, src], sv[f][None, dst])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            if const_expr(unroll):
                for j in cutlass.range(tile_n, unroll=unroll):
                    acc = combine_fn(acc, take(j))
            else:
                for j in cutlass.range_constexpr(tile_n):
                    acc = combine_fn(acc, take(j))
        return acc

    @cute.jit
    def _fold_itree_combine(self, mIns, row):
        """Fold split partials linearly from partial zero, grouping register loads."""
        # Local binding avoids trait attribute access inside the dynamic loop.
        combine_fn, fdtypes = self.trait.combine, self.trait.fdtypes
        nf = const_expr(self.trait.nfields)
        nbatch = const_expr(self.itree.split[0])
        grp = const_expr(self.itree.combine_grp)
        base = row * Int32(nbatch)
        if const_expr(grp == 1):
            pull = lambda i: tuple(fdtypes[f](mIns[f][i]) for f in range(nf))  # noqa: E731
            acc = pull(base)
            for b in cutlass.range(1, nbatch):
                acc = combine_fn(acc, pull(base + b))
            return acc
        frags = [
            cute.make_rmem_tensor(cute.make_layout(grp), mIns[f].element_type)
            for f in range(nf)
        ]
        take = lambda j: tuple(fdtypes[f](frags[f][j]) for f in range(nf))  # noqa: E731
        for f in cutlass.range_constexpr(nf):
            cute.autovec_copy(
                cute.make_tensor(mIns[f].iterator + base, cute.make_layout(grp)),
                frags[f],
            )
        acc = take(0)
        for j in cutlass.range_constexpr(grp - 1):
            acc = combine_fn(acc, take(const_expr(j + 1)))
        for g in cutlass.range(1, nbatch // grp):
            off = base + Int32(grp) * Int32(g)
            for f in cutlass.range_constexpr(nf):
                cute.autovec_copy(
                    cute.make_tensor(mIns[f].iterator + off, cute.make_layout(grp)),
                    frags[f],
                )
            for j in cutlass.range_constexpr(grp):
                acc = combine_fn(acc, take(j))
        return acc

    @cute.jit
    def _fold_partials(self, mIns, unit, nchunks, npar):
        # Fold one column's stage-2 partials. Local method bindings avoid dynamic-loop
        # attribute access, which trips the IR flattener.
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
        rexts,
        rstrides,
        kexts,
        kstrides,
        in_base,
        limit,
        stream,
    ):
        # Build and bake the TMA atom at host compile time, replacing the input with its descriptor.
        if const_expr(self.use_tma):
            tma_atom, mTma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mIns[0],
                smem_box_layout(self.storage_width, self.threads_per_block),
                self.tiler,
            )
            mIns = [mTma]
        else:
            tma_atom = None
        if const_expr(self.order == "inner_tree"):
            # Split inner-tree writes one partial per (output, batch).
            nout = mOuts[0].shape[0] // const_expr(self.output_widths[0])
            gx = cute.ceil_div(nout, const_expr(self.rows_per_block))
            gy = Int32(1)
        elif const_expr(self.axis == "general"):
            if const_expr(self.wide_output):
                # Host geometry splits output blocks over CUDA's signed 32-bit grid x.
                gx = q
                gy = npar
            else:
                gx = mOuts[0].shape[0] // const_expr(self.output_widths[0])
                gy = Int32(1)
        elif const_expr(self.axis == "row"):
            nout = mIns[0].shape[0]
            gx = cute.ceil_div(nout, const_expr(self.rows_per_block))
            gy = Int32(1)
        else:
            gx = cute.ceil_div(nchunks, const_expr(self.threads_per_block))
            # Grid y splits stage 1's reduced axis; combine has consumed it.
            gy = Int32(1) if const_expr(self.combine) else npar
        # Build V2 divisors in the MLIR context so .divisor crosses the kernel boundary.
        rdivs = (
            rexts
            if const_expr(self.wide_red)
            else (
                [cute.FastDivmodDivisorV2(e) for e in rexts]
                if rexts is not None
                else None
            )
        )
        kdivs = (
            kexts
            if const_expr(self.wide_kept)
            else (
                [cute.FastDivmodDivisorV2(e) for e in kexts]
                if kexts is not None
                else None
            )
        )
        self.kernel(
            mIns,
            mOuts,
            tma_atom,
            nchunks,
            nwaves,
            project_n,
            q,
            npar,
            rdivs,
            rstrides,
            kdivs,
            kstrides,
            in_base,
            limit,
        ).launch(
            grid=[gx, gy, 1],
            block=[const_expr(self.threads_per_block), 1, 1],
            stream=stream,
        )

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
        rdivs,
        rstrides,
        kdivs,
        kstrides,
        in_base,
        limit,
    ):
        # Runtime geometry shares kernels across extents. Pass None for unused arguments;
        # one extra Int32 slowed column fold 1.27x (8.2 to 10.4us at (65536, 256)).
        tx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        grid_x = cute.arch.grid_dim()[0] if const_expr(self.wide_output) else Int32(0)
        trait = self.trait
        chunk_base = (
            Int64(0) if const_expr(self.wide_gidx) else Int32(0)
        )  # nonzero only under ragged_chunk, for gidx_from "chunk"
        batch_idx = None  # split shape only: which batch of the row this block folds
        # Inner-tree thread map, set from its plan.
        lane_w = warp_id = row_in_block = None
        if const_expr(self.order == "inner_tree"):
            # wpr warps cooperate per row; rows_per_block groups share the block.
            wpr = const_expr(self.itree.wpr)
            if const_expr(wpr == 0):
                # One thread owns and stores each row without merging.
                lane_w, warp_id, row_in_block, lane = (Int32(0),) * 4
            elif const_expr(self.itree.stage_e):
                # One staged warp per row needs no chunk index or cross-warp merge.
                lane_w = Int32(tx % WARP)
                lane = lane_w
                warp_id = Int32(0)
                row_in_block = Int32(tx // WARP)
            else:
                # warp_id selects this warp's group of kchunk adjacent chunks.
                groups = const_expr(wpr // self.itree.kchunk)
                lane_w = Int32(tx % WARP)
                warp = Int32(tx // WARP)
                row_in_block = warp // Int32(groups)
                warp_id = warp % Int32(groups)
                # Shared store uses lane 0 for the merged total.
                lane = Int32(tx % const_expr(WARP * groups))
            if const_expr(self.itree.shape == "split"):
                # Grid and output index pair each row with each batch.
                nb = const_expr(self.itree.split[0])
                pair = Int32(bx) * const_expr(self.itree.rows_per_block) + row_in_block
                raw = pair // Int32(nb)
                batch_idx = pair % Int32(nb)
                alive = True
            elif const_expr(self.itree.shape == "combine" and self.itree.combine_tile):
                raw = Int32(bx) * const_expr(self.itree.rows_per_block) + Int32(tx)
                alive = (Int32(tx) < const_expr(self.itree.rows_per_block)) & (
                    raw < Int32(mOuts[0].shape[0] // const_expr(self.output_widths[0]))
                )
            elif const_expr(wpr == 0):
                raw = Int32(bx) * const_expr(self.threads_per_block) + Int32(tx)
                alive = raw < Int32(
                    mOuts[0].shape[0] // const_expr(self.output_widths[0])
                )
            else:
                raw = Int32(bx) * const_expr(self.itree.rows_per_block) + row_in_block
                alive = raw < Int32(
                    mOuts[0].shape[0] // const_expr(self.output_widths[0])
                )
        elif const_expr(self.axis == "general"):
            # One block per output, folded by every thread.
            raw = (
                Int64(bx) + Int64(by) * Int64(grid_x)
                if const_expr(self.wide_output)
                else Int32(bx)
            )
            lane = Int32(tx)
            alive = (
                raw < Int64(mOuts[0].shape[0] // const_expr(self.output_widths[0]))
                if const_expr(self.wide_output)
                else True
            )
        elif const_expr(self.axis == "row"):
            raw = Int32(bx) * const_expr(self.rows_per_block) + Int32(
                tx // const_expr(self.threads_per_row)
            )
            lane = Int32(tx % const_expr(self.threads_per_row))
            alive = raw < Int32(mIns[0].shape[0])
        else:
            raw = Int32(bx) * const_expr(self.threads_per_block) + Int32(tx)
            lane = Int32(0)  # the col mapping gives every output group its own thread
            alive = raw < nchunks
        # Clamp dead threads to unit 0 and discard them at store, avoiding predicated loads.
        unit = (
            raw
            if alive
            else (
                Int64(0)
                if const_expr(self.axis == "general" and self.wide_output)
                else Int32(0)
            )
        )

        if const_expr(self.order == "inner_tree"):
            obase = in_base
            if const_expr(self.axis == "general" and self.npairs_kept > 0):
                obase = in_base + _decode_offset(
                    unit, kdivs, kstrides, const_expr(self.npairs_kept)
                )
            if const_expr(self.itree.shape == "combine"):
                if const_expr(self.itree.combine_tile):
                    accs = (
                        self._fold_itree_combine_async(
                            mIns, unit, Int32(tx), Int32(bx)
                        ),
                    )
                else:
                    accs = (self._fold_itree_combine(mIns, unit),)
            elif const_expr(self.itree.stage_e):
                accs = (
                    self._fold_itree_smem(
                        mIns[0], unit, lane_w, row_in_block, batch_idx
                    ),
                )
            elif const_expr(self.itree.stage_rows):
                # With one thread per row the row within the block IS the thread index; the mapping above
                # zeroes it because the direct fold indexes the whole tensor, and the staged tile does not.
                rib = Int32(tx) if const_expr(self.itree.wpr == 0) else row_in_block
                accs = (
                    self._fold_itree_rowstage(
                        mIns[0], Int32(tx), Int32(bx), rib, lane_w, warp_id, batch_idx
                    ),
                )
            else:
                accs = (
                    self._fold_itree(
                        mIns[0],
                        unit,
                        lane_w,
                        warp_id,
                        row_in_block,
                        batch_idx,
                        obase,
                        rdivs,
                        rstrides,
                        in_base,
                    ),
                )
        elif const_expr(self.axis == "general"):
            # Decode this output's base; reduce-all with no kept pairs uses in_base.
            obase = in_base
            if const_expr(self.npairs_kept > 0):
                obase = in_base + _decode_offset(
                    unit,
                    kdivs,
                    kstrides,
                    const_expr(self.npairs_kept),
                    const_expr(self.wide_kept),
                )
            rb = nchunks
            if const_expr(self.flat_tail):
                # Clamp reduce-all stage 1's overhanging final chunk.
                chunk_base = (
                    Int64(unit) * Int64(nchunks)
                    if const_expr(self.wide_gidx)
                    else unit * nchunks
                )
                left = limit - obase
                c64 = cutlass.Int64(nchunks)
                left = left if left < c64 else c64  # noqa: FURB136 -- no DSL builtin min
                zero = cutlass.Int64(0)
                left = left if left > zero else zero  # noqa: FURB136 -- no builtin max
                rb = (
                    cutlass.Int64(left)
                    if const_expr(self.wide_count)
                    else cutlass.Int32(left)
                )
            elif const_expr(self.ragged_chunk):
                # Clamp each short final chunk. The fastest kept pair gives its step index
                # for either row or column splitting.
                _, cc = divmod(unit, kdivs[0])
                c = cutlass.Int64(cc)
                cnt = cutlass.Int64(nchunks)
                chunk_base = (
                    c * cnt if const_expr(self.wide_gidx) else Int32(c * cnt)
                )  # this chunk's first step, for gidx
                left = limit - c * cnt
                c64 = cutlass.Int64(nchunks)
                left = left if left < c64 else c64  # noqa: FURB136 -- no builtin min
                zero = cutlass.Int64(0)
                left = left if left > zero else zero  # noqa: FURB136 -- no builtin max
                rb = (
                    cutlass.Int64(left)
                    if const_expr(self.wide_count)
                    else cutlass.Int32(left)
                )
            if const_expr(self.combine):
                accs = (
                    fold_partials_run(
                        trait,
                        mIns,
                        obase,
                        rb,
                        const_expr(self.threads_per_block),
                        Int64(lane) if const_expr(self.wide_count) else lane,
                        in_base,
                    ),
                )
            elif const_expr(self.general_tree_count):
                accs = (
                    fold_decoded_uniform_tree(
                        trait,
                        mIns[0],
                        obase,
                        rstrides,
                        const_expr(self.threads_per_block),
                        lane,
                        const_expr(self.general_tree_count),
                    ),
                )
            else:
                accs = (
                    fold_decoded(
                        trait,
                        mIns[0],
                        obase,
                        rdivs,
                        rstrides,
                        const_expr(self.npairs_red),
                        rb,
                        const_expr(self.threads_per_block),
                        Int64(lane) if const_expr(self.wide_count) else lane,
                        in_base,
                        chunk_base,
                        const_expr(self.gidx_from),
                        const_expr(self.wide_gidx),
                        const_expr(self.wide_red),
                        const_expr(self.complex_input),
                    ),
                )
            accs = (merge_lanes(trait, accs[0], const_expr(self.threads_per_row)),)
            accs = (self._block_merge(accs[0]),)
        elif const_expr(self.combine):
            accs = (self._fold_partials(mIns, unit, nchunks, npar),)
        elif const_expr(self.axis == "col"):
            # Clamp this block's possibly ragged reduced-axis chunk.
            index_row0 = Int32(by) * q
            left = project_n - index_row0
            cnt = left if left < q else q  # noqa: FURB136 -- no DSL builtin min
            # A capped npar may leave blocks empty; clamp explicitly instead of relying on
            # negative trip counts lowering to zero.
            zero = Int32(0)
            cnt = cnt if cnt > zero else zero  # noqa: FURB136 -- no DSL builtin max
            if const_expr(self.batched_col):
                batch = unit // nwaves
                col = unit % nwaves
                storage_row0 = batch * project_n + index_row0
            else:
                col = unit
                storage_row0 = index_row0
            if const_expr(self.col_group > 1):
                accs = fold_col_welford_grouped(
                    trait,
                    mIns[0],
                    col,
                    storage_row0,
                    index_row0,
                    cnt,
                    const_expr(self.col_group),
                )
            else:
                accs = fold_cols_rolled(
                    trait,
                    mIns[0],
                    col,
                    storage_row0,
                    index_row0,
                    cnt,
                    const_expr(self.vec),
                    const_expr(self.unroll),
                    const_expr(self.complex_input),
                )
        elif const_expr(self.use_tma):
            accs = (self._fold_tma(mIns[0], tma_atom, bx, tx),)
        elif const_expr(self.ragged_vector):
            acc = fold_row_aligned(
                trait,
                mIns[0],
                unit,
                self.tm,
                lane,
                Int32(const_expr(self.N)),
                const_expr(self.unroll),
            )
            acc = merge_lanes(trait, acc, const_expr(self.threads_per_row))
            accs = (self._block_merge(acc),)
        elif const_expr(self.threads_per_row == 1):
            # One thread owns the row, so omit wave/lane arithmetic and its predicate.
            accs = (
                fold_linear_rolled(
                    trait,
                    mIns[0],
                    unit,
                    const_expr(self.vec),
                    nchunks,
                    const_expr(self.unroll),
                    const_expr(self.complex_input),
                ),
            )
        else:
            if const_expr(self.row_accumulators):
                acc = fold_row_static(
                    trait,
                    mIns[0],
                    unit,
                    self.tm,
                    lane,
                    const_expr(self.row_accumulators),
                )
            elif const_expr(self.load_ahead == 4):
                acc = fold_row_prefetched4(
                    trait,
                    mIns[0],
                    unit,
                    self.tm,
                    lane,
                    nchunks,
                    nwaves,
                    const_expr(self.complex_input),
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
                    const_expr(self.complex_input),
                )
            acc = merge_lanes(trait, acc, const_expr(self.threads_per_row))
            accs = (self._block_merge(acc),)

        # Compute output indices after folding to reduce live registers.
        out_base = unit * const_expr(self.nslots)
        if const_expr(self.axis in ("row", "general")):
            # Split inner-tree writes by block; other row/general paths write by unit.
            part_base = (
                Int32(bx) * const_expr(self.itree.rows_per_block) + row_in_block
                if const_expr(
                    self.order == "inner_tree" and self.itree.shape == "split"
                )
                else unit
            )
            part_stride = Int32(1)
        else:
            # COL: (P, C) stores this chunk's columns in row `by`; (C, P) interleaves
            # partials per column for block-per-column combination.
            part_base = (
                Int32(by) * (nchunks * const_expr(self.nslots)) + out_base
                if const_expr(self.pc)
                else out_base * npar + Int32(by)
            )
            part_stride = Int32(1) if const_expr(self.pc) else npar

        # Project before the dynamic store branch; binding there is rejected and leaks the trait.
        if const_expr(self.final):
            res = tuple(trait.project(a, trait.acc(project_n)) for a in accs)
            if lane == 0 and alive:
                for s in cutlass.range_constexpr(self.nslots):
                    if const_expr(self.nouts == 1):
                        _store_reduction_value(
                            mOuts[0],
                            out_base + Int32(s),
                            res[s],
                            const_expr(self.output_widths[0]),
                        )
                    else:
                        for k in cutlass.range_constexpr(self.nouts):
                            _store_reduction_value(
                                mOuts[k],
                                out_base + Int32(s),
                                res[s][k],
                                const_expr(self.output_widths[k]),
                            )
        else:
            # Emit raw per-field partials for stage 2.
            if lane == 0 and alive:
                for s in cutlass.range_constexpr(self.nslots):
                    for f in cutlass.range_constexpr(trait.nfields):
                        mOuts[f][part_base + Int32(s) * part_stride] = trait.fdtypes[f](
                            accs[s][f]
                        )
