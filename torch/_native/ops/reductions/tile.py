# Shared reduction kernel/datapath; drivers own launch and cache policy. Reuse avoids
# prior 3.7x unwidened and 3x undeclared-alignment regressions. Tiles map vec
# elements/load across tpr threads/row; rolled folds share vector-class kernels.

import math

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
    from .traits import warp_reduce

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
    ):
        if axis not in ("row", "col"):
            raise ValueError(f"axis must be 'row' or 'col', got {axis!r}")
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
        itemsize = dtype.width // 8
        # Runtime row folds need one mapped load; TMA declares its static staged depth.
        # Column vec is a driver choice, so columns need no tile.
        self.tm = (
            TileMap(N, itemsize, tpr, N // vec_size(N, itemsize) if use_tma else 1)
            if axis == "row"
            else None
        )
        if axis == "row":
            self.vec = self.tilemap.vec
        elif vec is None:
            # Column vec cannot be inferred from load width.
            raise ValueError("the col axis needs an explicit vec")
        else:
            self.vec = vec
        # Rows merge to one slot; columns keep vec independent slots.
        self.nslots = 1 if axis == "row" else self.vec
        self.rows_per_block = nt // tpr
        self.warps_per_row = tpr // WARP  # 0 at tpr == 1: nothing to merge
        self.tiler = (nt, N)  # TMA box: nt whole rows

    @property
    def tilemap(self):
        # Only row folds map a whole row onto one tile.
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
        self, mIns: list, mOuts: list, nchunks, nwaves, project_n, q, npar, stream
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
        if const_expr(self.axis == "row"):
            gx = cute.ceil_div(mIns[0].shape[0], const_expr(self.rows_per_block))
            gy = Int32(1)
        else:
            gx = cute.ceil_div(nchunks, const_expr(self.nt))
            # Grid y splits stage 1's reduced axis; combine has consumed it.
            gy = Int32(1) if const_expr(self.combine) else npar
        self.kernel(mIns, mOuts, tma_atom, nchunks, nwaves, project_n, q, npar).launch(
            grid=[gx, gy, 1], block=[const_expr(self.nt), 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self, mIns: list, mOuts: list, tma_atom, nchunks, nwaves, project_n, q, npar
    ):
        # Runtime geometry shares kernels across extents. Pass None for unused arguments;
        # one extra Int32 slowed column fold 1.27x (8.2 to 10.4us at (65536, 256)).
        tx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        trait = self.trait
        if const_expr(self.axis == "row"):
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

        if const_expr(self.combine):
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
            acc = merge_lanes(trait, acc, self.tm)
            if const_expr(self.warps_per_row > 1):
                from .traits import block_reduce

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
            accs = (acc,)

        # Compute output indices after folding to reduce live registers.
        out_base = unit * const_expr(self.nslots)
        if const_expr(self.axis == "row"):
            part_base = unit
            part_stride = Int32(1)
        else:
            # (P, C) stores this chunk's columns in row `by`; (C, P) interleaves
            # partials per column.
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
