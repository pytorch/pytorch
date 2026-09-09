# The shared reduction KERNEL and its datapath: where a tile's load width, alignment and thread
# mapping are derived, the folds that walk them, and the ONE @cute.kernel every fast reduction
# path launches. The kernel_* modules are drivers -- they pick the launch shape and own the plan
# cache -- but the body lives here.

import math

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64, pipeline
from cutlass.cute.nvgpu import cpasync


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


def _decode_offset(linear, divs, strides, npairs):
    # Mixed-radix decode of a linear index to a flat element offset; only the pair COUNT is baked,
    # so one kernel serves every geometry sharing it. The div/mod stays INT32 -- `linear` spans
    # count or num_o, both asserted < 2**31 -- and only the offset product needs Int64. npairs >= 1:
    # the caller drops the decode for an empty KEPT list.
    rem = Int32(linear)
    if npairs == 1:
        return cutlass.Int64(rem) * strides[0]
    off = cutlass.Int64(0)
    for j in range(npairs - 1):
        q, r = divmod(rem, divs[j])
        off = off + cutlass.Int64(r) * strides[j]
        rem = q
    return off + cutlass.Int64(rem) * strides[npairs - 1]


class TileMap:
    """How one row is spread over threads and loads. tpr == 1 means one thread owns a whole row,
    with no lane merge.
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
    """Grid-stride fold of ONE output's reduced run, addressed by mixed-radix DECODE.

    The arbitrary-layout arm: a transposed or expanded input is a different decode, not a
    different kernel. All `nt` threads cooperate on one output and the caller merges them.
    `gidx` picks what an index trait is told the position is.
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
    # Invalid lanes read in_base (always in range): obase itself can be past the end for an
    # overhanging chunk, whose rb clamps to 0.
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
    """COMBINE one output's pre-reduced accumulator tuples, grid-strided by `nt`. The stage-2
    reader for every per-output split; the loop is dynamic because the count reaches ~1e5.
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
    """Reduce across the `tpr` lanes covering one row; a no-op at tpr == 1.

    `asc` selects the ASCENDING butterfly, which the folds' column order depends on and an
    index trait's ties depend on.
    """
    if const_expr(tpr == 1):
        return acc
    from .._cutedsl.traits import warp_reduce

    return warp_reduce(trait, acc, tpr, ascending=asc)


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


def smem_box_layout(N: int, threads: int):
    """Smem destination for a (threads, N) TMA box of whole rows: plain row-major.

    The bank conflict is dealt with in the ACCESS PATTERN instead (see fold_smem_rotated):
    TMA accepts only the GEMM swizzles, which do not de-conflict a whole-row read.
    """
    return cute.make_ordered_layout((threads, N), order=(1, 0))


@cute.jit
def fold_smem_rotated(trait, sX, rb, N: cutlass.Constexpr):
    """Fold row `rb` of a staged (threads, N) smem tile. One thread per row, no lane merge.

    Indexed logically so the true column reaches the trait, and ROTATED by row index: thread t
    reads row t, so an unrotated read puts every lane in one bank -- a 32-way conflict costing
    more than TMA's coalescing buys. N is a power of two, so the modulo is a mask.
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

    The transpose of every other fold here: vectorized along the KEPT axis, so a thread owns
    adjacent columns with one accumulator each and never merges across lanes.
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
        # Plain `range`: a comprehension is not visited by the DSL AST preprocessor, so
        # range_constexpr raises there -- and vec is compile-time, so this unrolls the same way.
        accs = tuple(reduce_fn(accs[i], acc_dt(frag[i]), rr, True) for i in range(vec))
    return accs


class TileReduce:
    """The tile reduction KERNEL: one body, parameterized by which axis is reduced.

    Row: the reduced axis is CONTIGUOUS, `tpr` threads share a row and then merge (tpr == 1
    merges nothing). Col: the reduced axis is STRIDED, each thread owns `vec` adjacent columns
    and folds down the rows, so nothing merges across lanes and the y-grid splits the reduced
    axis instead.

    One body and not two because only the FOLD is axis-specific: the dead-thread clamp, the
    projection and the store are shared, and both axes write nslots x nouts results.
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
    ):
        if axis not in ("row", "col", "general"):
            raise ValueError(f"axis must be 'row', 'col' or 'general', got {axis!r}")
        if axis == "general":
            # Every thread of the block folds the one output, so the tail is the row axis's
            # at tpr == nt.
            tpr = nt
            if nt % WARP:
                # _block_merge derives warps_per_row from nt, so a block that is not a whole number of warps
                # leaves the last partial warp OUT of the merge -- a silently wrong reduction, measured 12-62%
                # low. `block` is caller-settable, so the row axis's own check is not enough.
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
            # The rotation is `& (N - 1)`, a rotation only at a power-of-two N: at any other N it
            # duplicates some columns and skips others, returning a plausible number. tma_ok declines
            # those, but use_tma is caller-settable and bypasses it.
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
        itemsize = dtype.width // 8 if dtype is not None else 0
        # The row folds take their tile from TileMap with loads=1 because they are ROLLED, so the
        # static count is unused and the unroll bound is trivially met. The TMA fold walks a staged
        # row with a compile-time trip count, so it declares its real depth. The col axis needs no
        # tile (its vec is a driver choice), and the fixed-DAG order carries one per batch.
        self.tm = (
            TileMap(N, itemsize, tpr, N // vec_size(N, itemsize) if use_tma else 1)
            if axis == "row"
            else None
        )
        if axis == "row":
            self.vec = self.tilemap.vec
        elif axis == "general":
            # the general axis loads one element at a time (an arbitrary stride pattern has no
            # width to exploit), so it has no tile and no vector
            self.vec = 1
        elif vec is None:
            # The col axis takes `vec` from its DRIVER (accumulators per thread, not a load
            # width), so there is nothing here to derive it from.
            raise ValueError("the col axis needs an explicit vec")
        else:
            self.vec = vec
        # one output per thread on the row axis (its lanes are merged first); `vec` adjacent
        # columns, each with its own accumulator, on the col axis
        self.nslots = self.vec if axis == "col" else 1
        self.rows_per_block = nt // tpr
        self.warps_per_row = tpr // WARP  # 0 at tpr == 1: nothing to merge
        self.tiler = (nt, N)  # TMA box: nt whole rows

    @property
    def tilemap(self):
        # Set only for a fold that takes ONE tile for the whole row. The col axis derives its
        # `vec` from the driver rather than from a load width, so it carries no tile at all.
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
        )

    @cute.jit
    def _fold_tma(self, mX, tma_atom, bx, tx):
        # Stage this block's box of WHOLE rows into smem with one descriptor-driven transfer, then
        # fold out of smem. See _TMA_MIN_STRIDE for why, and fold_smem_rotated for the rotation.
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
        # Warp 0 issues the transfer; narrowing it to a single THREAD deadlocks, because the pipeline
        # signals per warp-lane-0 and syncs the warp internally, so a whole warp must enter the
        # producer region. Rows past M are zero-filled and discarded at the guarded store.
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
        # Merge one output's per-warp accumulators through smem. A no-op unless the output spans more
        # than one warp, which is every general-axis launch of >= 64 threads.
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
    def _fold_partials(self, mIns, unit, nchunks, npar):
        # COMBINE pass (col axis stage 2): fold this thread's column's partials, which the split left
        # as one matrix per field. Bind the trait's methods to locals -- attribute access on it inside
        # a dynamic loop trips the IR flattener.
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
            gx = cute.ceil_div(mIns[0].shape[0], const_expr(self.rows_per_block))
            gy = Int32(1)
        else:
            gx = cute.ceil_div(nchunks, const_expr(self.nt))
            # the y-grid splits the REDUCED axis in stage 1; combine has already consumed it
            gy = Int32(1) if const_expr(self.combine) else npar
        # The divisors need an MLIR context, so they are built here rather than by the caller. V2
        # rather than V1 so `.divisor` stays readable across the kernel boundary.
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
        # RUNTIME args, so one compiled kernel serves every extent sharing a structure. An arg a
        # variant does not use is passed as None, not a dummy: one unused Int32 param measured 1.27x
        # on the column fold (8.2 -> 10.4us at (65536, 256)).
        tx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        trait = self.trait
        chunk_base = Int32(0)  # nonzero only under ragged_chunk, for gidx_from "chunk"
        if const_expr(self.axis == "general"):
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
        # Dead threads clamp onto unit 0 so every load stays in range; their accumulator is discarded
        # at the guarded store. Cheaper than predicating every load, and a fold has no side effects.
        unit = raw if alive else Int32(0)

        # THE FOLD: the one axis-specific step. Everything after it is shared.
        if const_expr(self.axis == "general"):
            # Base flat offset of this output (decode the block index against the kept dims;
            # zero kept pairs -- a full reduction -- leaves just in_base), then the fold bound.
            obase = in_base
            if const_expr(self.npairs_kept > 0):
                obase = in_base + _decode_offset(
                    unit, kdivs, kstrides, const_expr(self.npairs_kept)
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
                # A split whose chunk need not divide the reduced run: the LAST chunk of every output
                # is short and must not fold the next one's elements. The chunk pair is the
                # fastest-varying kept pair, so one divmod per BLOCK yields the chunk index. Counted
                # in STEPS, so a contiguous row split and a column split both use it unchanged.
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
            # This block's chunk of the REDUCED axis. The last chunk is short whenever q does not divide
            # the extent -- the same ragged tail the row split has, clamped the same way.
            row0 = Int32(by) * q
            left = project_n - row0
            cnt = left if left < q else q  # noqa: FURB136 -- no DSL builtin min
            # _split_p caps npar, so the last blocks can be left with nothing. A negative trip count
            # happens to lower to a zero-trip loop, which is not a guarantee to rest a global load on.
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
            # The general axis pins gy to 1, so the col arm below would come out as this anyway through a
            # runtime multiply that is always zero.
            part_base = unit
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
