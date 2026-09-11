# Shared reduction datapath; drivers own launch/cache policy. Tiles map vec
# elements/load across tpr threads/row. Rolled folds share vector-class kernels;
# fixed DAGs preserve order. Reuse avoids prior 3.7x unwidened and 3x alignment losses.

import math
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64, pipeline
from cutlass.cute.nvgpu import cpasync


WARP = 32

# Static unroll compiles superlinearly past ~1300 operations.
MAX_UNROLL = 512


def vec_size(N: int, itemsize: int) -> int:
    """Elements/load; gcd makes vec divide N and preserves alignment at every chunk."""
    return math.gcd(N, max(1, 16 // itemsize))


def align_bytes(N: int, itemsize: int) -> int:
    """Input alignment declaration; omitting it emits narrow loads and costs 3x."""
    return vec_size(N, itemsize) * itemsize


def _decode_offset(linear, divs, strides, npairs):
    # Mixed-radix flat offset with only pair count compiled. linear < 2**31 keeps
    # div/mod Int32; stride products use Int64. Callers omit empty decodes.
    rem = Int32(linear)
    if npairs == 1:
        return cutlass.Int64(rem) * strides[0]
    off = cutlass.Int64(0)
    for j in range(npairs - 1):
        q, r = divmod(rem, divs[j])
        off = off + cutlass.Int64(r) * strides[j]
        rem = q
    return off + cutlass.Int64(rem) * strides[npairs - 1]


def _off(base, i: int):
    # Keep a static base static (see TileMap.col_base): int + int stays foldable.
    return base + i if isinstance(base, int) else base + Int32(i)


class TileMap:
    """Map row elements to threads and loads; tpr=1 needs no lane merge."""

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
        # Cross-warp butterflies require a power-of-two warp count or drop partials.
        nw = tpr // WARP
        if tpr != 1 and (tpr % WARP or nw & (nw - 1)):
            raise ValueError(
                f"tpr must be 1 or a power-of-two multiple of {WARP}, got {tpr}"
            )
        unroll = (vec_size(N, itemsize) if vec is None else vec) * loads
        if unroll > MAX_UNROLL:
            raise ValueError(
                f"per-thread unroll {unroll} (vec*loads) exceeds MAX_UNROLL={MAX_UNROLL}; "
                f"compile time scales with it (~1 ms/op, superlinear past ~1300). "
                f"Got N={N} tpr={tpr} loads={loads}."
            )
        self.N = N
        # Fixed DAGs pass an N-independent vec; deriving it from N would change bits.
        self.vec = vec_size(N, itemsize) if vec is None else vec
        self.tpr = tpr
        self.loads = loads
        self.warp_major = warp_major
        self.nw = 1 if tpr == 1 else tpr // WARP
        # Exact full-row tiles need no load predicates; batched tiles are never exact.
        self.exact = (self.vec * self.loads * self.tpr == N) if exact is None else exact
        # vec must divide N to align every row and chunk; otherwise load by element.
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
        """(lane, warp, load) strides; warp_major swaps the warp/load assignment."""
        if self.tpr == 1:
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
        if self.tpr == 1:
            return l * s_l
        return lane * Int32(s_lane) + w * s_w + Int32(const_expr(l * s_l))


@cute.jit
def fold_decoded(
    trait,
    mX,
    obase,
    rdivs,
    rstrides,
    npairs: cutlass.Constexpr,
    rb,
    nt: cutlass.Constexpr,
    tidx,
    in_base,
    chunk_base,
    gidx: cutlass.Constexpr = "r",
):
    """Grid-stride one output using mixed-radix addressing.

    Runtime decodes cover arbitrary layouts; nt threads cooperate and gidx selects
    the index reported to traits.
    """
    reduce_fn, acc_dt = trait.reduce, trait.acc
    acc = trait.init()
    n_full = rb // nt
    base_r = tidx
    for _ in cutlass.range(n_full):
        # Inline offsets to avoid a spurious loop-carried value.
        if const_expr(gidx == "flat"):
            acc = reduce_fn(
                acc,
                acc_dt(mX[obase + _decode_offset(base_r, rdivs, rstrides, npairs)]),
                Int32(obase + _decode_offset(base_r, rdivs, rstrides, npairs)),
                True,
            )
        elif const_expr(gidx == "chunk"):
            acc = reduce_fn(
                acc,
                acc_dt(mX[obase + _decode_offset(base_r, rdivs, rstrides, npairs)]),
                chunk_base + base_r,
                True,
            )
        else:
            acc = reduce_fn(
                acc,
                acc_dt(mX[obase + _decode_offset(base_r, rdivs, rstrides, npairs)]),
                base_r,
                True,
            )
        base_r = base_r + nt
    # Invalid lanes read in_base because overhanging chunks may put obase out of range.
    valid = base_r < rb
    off = obase + _decode_offset(base_r, rdivs, rstrides, npairs)
    off_s = off if valid else in_base
    val = acc_dt(mX[off_s])
    if const_expr(gidx == "flat"):
        return reduce_fn(acc, val, Int32(off_s), valid)
    if const_expr(gidx == "chunk"):
        return reduce_fn(acc, val, chunk_base + base_r, valid)
    return reduce_fn(acc, val, base_r, valid)


@cute.jit
def fold_partials_run(trait, mIns, obase, rb, nt: cutlass.Constexpr, tidx, in_base):
    """Grid-stride one output's partial tuples for stage 2.
    The loop stays dynamic because partial counts reach about 1e5.
    """
    combine_fn = trait.combine
    fdtypes = trait.fdtypes
    nf = const_expr(trait.nfields)
    acc = trait.init()
    n_full = rb // nt
    r = tidx
    for _ in cutlass.range(n_full):
        rr = obase + cutlass.Int64(r)
        # Bare range unrolls and keeps trait attribute access outside the dynamic loop.
        acc = combine_fn(acc, tuple(fdtypes[f](mIns[f][rr]) for f in range(nf)))
        r = r + nt
    valid = r < rb
    rr = (obase + cutlass.Int64(r)) if valid else in_base
    part = tuple(fdtypes[f](mIns[f][rr]) for f in range(nf))
    merged = combine_fn(acc, part)
    return tuple((merged[f] if valid else acc[f]) for f in range(nf))


@cute.jit
def merge_lanes(trait, acc, tpr: cutlass.Constexpr, asc: cutlass.Constexpr = False):
    """Merge row lanes; `asc` controls numeric association and index ties. No-op at tpr=1."""
    if const_expr(tpr == 1):
        return acc
    from .traits import warp_reduce

    return warp_reduce(trait, acc, tpr, ascending=asc)


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
            x = trait.leaf(frag[j, i], col)
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
    for l in cutlass.range_constexpr(tm.loads):
        # Plain addition preserves a static base and promotes a dynamic one.
        base = tm.col_base(lane, w, l, warp_stride) + base_col
        if const_expr(whole_row and tm.wide_ok):
            _wide(rowv, base, tm.vec, frag[None, l])
        elif const_expr(not tm.wide_ok):
            for i in cutlass.range_constexpr(tm.vec):
                # Inline because the DSL rejects binding a dynamic value in a dynamic branch.
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
    """Plain row-major smem for a (threads, N) TMA box.

    TMA's GEMM swizzles do not fix whole-row bank conflicts; fold_smem_rotated does.
    """
    return cute.make_ordered_layout((threads, N), order=(1, 0))


@cute.jit
def fold_smem_rotated(trait, sX, rb, N: cutlass.Constexpr):
    """Fold one staged row per thread, rotating reads by row to avoid 32-way bank conflicts.

    Pass logical columns to the trait; power-of-two N makes rotation a mask.
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
    """Fold rows into vec kept-axis accumulators without lane merging."""
    reduce_fn, acc_dt = trait.reduce, trait.acc
    accs = tuple(trait.init() for _ in range(vec))
    frag = cute.make_rmem_tensor(cute.make_layout(vec), mX.element_type)
    for r in cutlass.range(nrows, unroll=unroll):
        # row0 selects this block's reduced-axis chunk.
        rr = row0 + Int32(r)
        cute.autovec_copy(
            cute.flat_divide(mX[Int64(rr), None], (vec,))[None, col], frag
        )
        # The DSL does not preprocess comprehensions; plain range still unrolls constexpr vec.
        accs = tuple(reduce_fn(accs[i], acc_dt(frag[i]), rr, True) for i in range(vec))
    return accs


class TileReduce:
    """Reduction kernel parameterized by reduced axis.

    Row uses tpr threads per contiguous row and merges lanes. Column gives each thread
    vec adjacent outputs and splits strided rows on grid y. Only folding is axis-specific;
    dead-thread handling, projection, and nslots-by-nouts stores are shared.
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
        vec: int | None = None,
        use_tma=False,
        combine=False,
        pc=True,
        npairs_red=0,
        npairs_kept=0,
        gidx_from="r",
        flat_tail=False,
        ragged_chunk=False,
        order="linear",
        # Duck-typed because its driver-owned type would invert the dependency.
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
            # The plan maps wpr chunks to wpr // kchunk warps and rows_per_block rows.
            tpr = WARP * (itree.wpr // itree.kchunk) if itree.wpr else 1
            nt = tpr * itree.rows_per_block
        if axis == "general":
            # Every block thread folds one output, equivalent to row tpr=nt.
            tpr = nt
            nwg = nt // WARP
            if nt % WARP or nwg & (nwg - 1):
                # Non-power-of-two warp counts omit partials in _block_merge, measuring
                # 12-62% low; caller-set block requires validation here.
                raise ValueError(
                    f"a general-axis block must be a power-of-two warp count, got {nt=}"
                )
        nwpr = tpr // WARP
        if (
            axis == "row"
            and tpr != 1
            and (tpr % WARP or tpr > nt or nt % tpr or nwpr & (nwpr - 1))
        ):
            raise ValueError(
                f"tpr must be 1 or a power-of-two multiple of {WARP} dividing nt: {tpr=} {nt=}"
            )
        if use_tma and (axis != "row" or tpr != 1):
            raise ValueError(
                f"TMA stages whole rows: needs row at tpr 1, {axis=} {tpr=}"
            )
        if use_tma and N & (N - 1):
            # The rotation mask duplicates and skips columns unless N is a power of two;
            # caller-set use_tma can bypass tma_ok.
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
        # Compile-time fold_decoded policy.
        self.npairs_red = npairs_red
        self.npairs_kept = npairs_kept
        self.gidx_from = gidx_from
        self.flat_tail = flat_tail
        self.ragged_chunk = ragged_chunk
        self.order = order
        self.itree = itree
        itemsize = dtype.width // 8 if dtype is not None else 0
        # Runtime row folds map one load; TMA declares static staged depth. Column and
        # general axes choose no tile.
        self.tm = (
            TileMap(N, itemsize, tpr, N // vec_size(N, itemsize) if use_tma else 1)
            if axis == "row" and order == "linear"
            else None
        )
        if axis == "row":
            self.vec = self.tilemap.vec if order == "linear" else itree.vec
        elif axis == "general":
            # Arbitrary strides offer no vector width.
            self.vec = 1
        elif vec is None:
            # Column vec cannot be inferred from load width.
            raise ValueError("the col axis needs an explicit vec")
        else:
            self.vec = vec
        # Columns keep vec slots; merged row and general axes keep one.
        self.nslots = self.vec if axis == "col" else 1
        self.rows_per_block = nt // tpr
        self.warps_per_row = tpr // WARP  # 0 at tpr == 1: nothing to merge
        self.tiler = (nt, N)  # TMA box: nt whole rows

    @property
    def tilemap(self):
        # Only linear row folds own one tile; inner-tree plans own one per batch.
        if self.tm is None:
            raise AssertionError(f"no tile on the {self.axis} axis")
        return self.tm

    @property
    def cache_sig(self):
        # Only TMA keys N because its box is static; runtime paths share a vector class.
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
            # Fixed DAGs key on N, requiring one kernel per shape instead of per vec class.
            self.itree.sig if self.itree is not None else None,
        )

    @cute.jit
    def _fold_tma(self, mX, tma_atom, bx, tx):
        # Transfer whole rows to smem once, then fold with rotated reads.
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
        # All of warp 0 must enter: the pipeline signals on lane 0 and synchronizes the warp.
        # One producer thread deadlocks. Rows past M are zero-filled and discarded.
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
    def _fold_itree(self, mX, r, lane_w, warp_id, row_in_block, batch_idx=None):
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
                frag = make_fragment(mX, tm)
                load(mX, r, tm, lane_w, cid, frag, base, bound, wstride)
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
    def _fold_itree_combine(self, mIns, row):
        """Fold each field's split partials linearly from 0 with a runtime count."""
        # Local binding avoids trait attribute access inside the dynamic loop.
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
                smem_box_layout(self.N, self.nt),
                self.tiler,
            )
            mIns = [mTma]
        else:
            tma_atom = None
        if const_expr(self.axis == "general"):
            # Read one-block-per-output grid live to serve any output count.
            gx = mOuts[0].shape[0]
            gy = Int32(1)
        elif const_expr(self.axis == "row"):
            # Split inner-tree writes (row, batch) partials, so count outputs, not input rows.
            nout = (
                mOuts[0].shape[0]
                if const_expr(self.order == "inner_tree")
                else mIns[0].shape[0]
            )
            gx = cute.ceil_div(nout, const_expr(self.rows_per_block))
            gy = Int32(1)
        else:
            gx = cute.ceil_div(nchunks, const_expr(self.nt))
            # Grid y splits stage 1's reduced axis; combine has consumed it.
            gy = Int32(1) if const_expr(self.combine) else npar
        # Build V2 divisors in the MLIR context so .divisor crosses the kernel boundary.
        rdivs = (
            [cute.FastDivmodDivisorV2(e) for e in rexts] if rexts is not None else None
        )
        kdivs = (
            [cute.FastDivmodDivisorV2(e) for e in kexts] if kexts is not None else None
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
        trait = self.trait
        chunk_base = Int32(0)  # nonzero only under ragged_chunk, for gidx_from "chunk"
        batch_idx = None  # split shape only: which batch of the row this block folds
        # Inner-tree thread map, set from its plan.
        lane_w = warp_id = row_in_block = None
        if const_expr(self.order == "inner_tree"):
            # wpr warps cooperate per row; rows_per_block groups share the block.
            wpr = const_expr(self.itree.wpr)
            if const_expr(wpr == 0):
                # One thread owns and stores each row without merging.
                lane_w, warp_id, row_in_block, lane = (Int32(0),) * 4
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
            # One block per output, folded by every thread.
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
        # Clamp dead threads to unit 0 and discard them at store, avoiding predicated loads.
        unit = raw if alive else Int32(0)

        if const_expr(self.order == "inner_tree"):
            if const_expr(self.itree.shape == "combine"):
                accs = (self._fold_itree_combine(mIns, unit),)
            else:
                accs = (
                    self._fold_itree(
                        mIns[0], unit, lane_w, warp_id, row_in_block, batch_idx
                    ),
                )
        elif const_expr(self.axis == "general"):
            # Decode this output's base; reduce-all with no kept pairs uses in_base.
            obase = in_base
            if const_expr(self.npairs_kept > 0):
                obase = in_base + _decode_offset(
                    unit, kdivs, kstrides, const_expr(self.npairs_kept)
                )
            rb = nchunks
            if const_expr(self.flat_tail):
                # Clamp reduce-all stage 1's overhanging final chunk.
                left = limit - obase
                c64 = cutlass.Int64(nchunks)
                left = left if left < c64 else c64  # noqa: FURB136 -- no DSL builtin min
                zero = cutlass.Int64(0)
                left = left if left > zero else zero  # noqa: FURB136 -- no builtin max
                rb = cutlass.Int32(left)
            elif const_expr(self.ragged_chunk):
                # Clamp each short final chunk. The fastest kept pair gives its step index
                # for either row or column splitting.
                _, cc = divmod(Int32(unit), kdivs[0])
                c = cutlass.Int64(cc)
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
                        rdivs,
                        rstrides,
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
            # Clamp this block's possibly ragged reduced-axis chunk.
            row0 = Int32(by) * q
            left = project_n - row0
            cnt = left if left < q else q  # noqa: FURB136 -- no DSL builtin min
            # A capped npar may leave blocks empty; clamp explicitly instead of relying on
            # negative trip counts lowering to zero.
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
            # One thread owns the row, so omit wave/lane arithmetic and its predicate.
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

        # Compute output indices after folding to reduce live registers.
        out_base = unit * const_expr(self.nslots)
        if const_expr(self.axis in ("row", "general")):
            # Split inner-tree writes by block; other row/general paths write by unit.
            part_base = (
                Int32(bx)
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
                        mOuts[0][out_base + Int32(s)] = mOuts[0].element_type(res[s])
                    else:
                        for k in cutlass.range_constexpr(self.nouts):
                            mOuts[k][out_base + Int32(s)] = mOuts[k].element_type(
                                res[s][k]
                            )
        else:
            # Emit raw per-field partials for stage 2.
            if lane == 0 and alive:
                for s in cutlass.range_constexpr(self.nslots):
                    for f in cutlass.range_constexpr(trait.nfields):
                        mOuts[f][part_base + Int32(s) * part_stride] = trait.fdtypes[f](
                            accs[s][f]
                        )
