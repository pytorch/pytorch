# The shared reduction KERNEL and its datapath: where a tile's load width, alignment and thread
# mapping are derived, the folds that walk them, and the ONE @cute.kernel every fast reduction
# path launches. The kernel_* modules are drivers -- they pick the launch shape and own the plan
# cache -- but the body lives here.

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
        combine=False,
        pc=True,
    ):
        if axis not in ("row", "col"):
            raise ValueError(f"axis must be 'row' or 'col', got {axis!r}")
        if axis == "row" and (tpr % WARP or tpr > nt or nt % tpr):
            raise ValueError(
                f"tpr must be a multiple of {WARP} dividing nt: {tpr=} {nt=}"
            )
        self.trait = trait
        self.dtype = dtype
        self.axis = axis
        self.N = N
        self.tpr = tpr
        self.nt = nt
        self.nouts = nouts
        self.final = final
        self.unroll = unroll
        self.combine = combine
        self.pc = pc  # partial layout: (P, C) when True, else (C, P)
        itemsize = dtype.width // 8
        # The row fold takes its tile from TileMap; loads=1 because it is ROLLED, so the
        # static per-thread count is unused and the MAX_UNROLL bound is trivially met. The col
        # axis needs no tile: its `vec` is a driver choice (accumulators per thread, not just
        # load width).
        self.tm = TileMap(N, itemsize, tpr, 1) if axis == "row" else None
        if axis == "row":
            self.vec = self.tilemap.vec
        elif vec is None:
            # The col axis takes `vec` from its DRIVER (accumulators per thread, not a load
            # width), so there is nothing here to derive it from.
            raise ValueError("the col axis needs an explicit vec")
        else:
            self.vec = vec
        # one output per thread on the row axis (its lanes are merged first); `vec` adjacent
        # columns, each with its own accumulator, on the col axis
        self.nslots = 1 if axis == "row" else self.vec
        self.rows_per_block = nt // tpr
        self.warps_per_row = tpr // WARP

    @property
    def tilemap(self):
        # Set only for a fold that takes ONE tile for the whole row. The col axis derives its
        # `vec` from the driver rather than from a load width, so it carries no tile at all.
        if self.tm is None:
            raise AssertionError(f"no tile on the {self.axis} axis")
        return self.tm

    @property
    def cache_sig(self):
        # N is ABSENT: every path takes its extents at runtime, so one compiled kernel serves
        # a whole vec class.
        return (
            self.axis,
            self.vec,
            self.tpr,
            self.nt,
            self.nouts,
            self.final,
            self.unroll,
            self.combine,
            self.pc,
            self.trait.nfields,
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
        self, mIns: list, mOuts: list, nchunks, nwaves, project_n, q, npar, stream
    ):
        if const_expr(self.axis == "row"):
            gx = cute.ceil_div(mIns[0].shape[0], const_expr(self.rows_per_block))
            gy = Int32(1)
        else:
            gx = cute.ceil_div(nchunks, const_expr(self.nt))
            # the y-grid splits the REDUCED axis in stage 1; combine has already consumed it
            gy = Int32(1) if const_expr(self.combine) else npar
        self.kernel(mIns, mOuts, nchunks, nwaves, project_n, q, npar).launch(
            grid=[gx, gy, 1], block=[const_expr(self.nt), 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mIns: list, mOuts: list, nchunks, nwaves, project_n, q, npar):
        # RUNTIME args, so one compiled kernel serves every extent sharing a structure. An arg a
        # variant does not use is passed as None, not a dummy: one unused Int32 param measured 1.27x
        # on the column fold (8.2 -> 10.4us at (65536, 256)).
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
        # Dead threads clamp onto unit 0 so every load stays in range; their accumulator is discarded
        # at the guarded store. Cheaper than predicating every load, and a fold has no side effects.
        unit = raw if alive else Int32(0)

        # THE FOLD: the one axis-specific step. Everything after it is shared.
        if const_expr(self.combine):
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
                from .._cutedsl.traits import block_reduce

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

        # Output indexing comes AFTER the fold: computed before it, these values stay live
        # across the loop and cost registers the fold wants.
        out_base = unit * const_expr(self.nslots)
        if const_expr(self.axis == "row"):
            part_base = unit
            part_stride = Int32(1)
        else:
            # (P, C) partials put this chunk's columns in row `by`; (C, P) interleaves them per
            # column, which is what a block-per-column stage 2 needs (see kernel_coltile).
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
