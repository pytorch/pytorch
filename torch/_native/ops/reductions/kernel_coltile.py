# Column-reduction driver for tile.TileReduce, with launch policy and plan cache, plus the
# PartialColumns stage-two kernel and the ordered (inner-tree) column kernel.
# TileReduce threads own vectorized kept-axis outputs without lane merging. Splitting the
# reduced axis provides parallelism; unsplit (65536, 256) took 7830us versus ATen's 15.8us.
# Ordered column lanes each serve one output, and a block's workers replay the row inner
# tree's warps, loading rows strided by C; worker 0 merges their roots.
# Inputs must be on the current device.

from collections.abc import Sequence
from typing import Any, Literal, NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64

import torch

from ...cutedsl import hw_caps as _hw, launch as _L
from ...cutedsl.dtypes import cute2torch, torch2cute
from ...cutedsl.plan_cache import cached_plan
from ...utils.device import check_current
from . import tile
from .kernel_general import (
    _B200_DTYPES,
    _combine_itree_partials,
    _flat,
    _INT32_LIMIT,
    _launch,
    _ORDERED_COL_CCS,
    _ORDERED_COL_DTYPES,
    ReduceBlock,
)
from .kernel_rowtile import trait_itree_plan
from .traits import WARP, welford_outputs


_compile = _L.compile_kernel
_stream = _L.stream
_CACHE = {}

# Stage one targets about _TARGET_LANES threads: (columns / vec) lanes per batch times
# npar reduced-axis chunks of at least _Q_MIN rows. vec narrows, trading width for lanes,
# until chunks keep _Q_VEC rows, and never leaves fewer than _MIN_LANES column lanes.
# Views of at most _SHORT_R rows reduce in one stage at vec <= 2. Against ceil(R / 64)
# chunks with vec <= 4, sum, argmax and var_mean (fp32/bf16; 1 and 64 MiB; C 16-4096;
# B 1 and 16) ran at 1.38x geomean on B200 and 1.33x on H100 (worst 0.75x), within 5%
# of the best (vec, npar) per shape.
_TARGET_LANES = 131072
_Q_MIN = 16
_Q_VEC = 64
_SHORT_R = 64
_MIN_LANES = 64
# Stage one chunks and ordered split partitions launch along grid y.
_MAX_NPAR = 65535
# Use blocks per column below the measured 4096-16384 crossover, then threads per column.
_C_THREAD_STAGE2 = 8192
# Small blocks avoid idle threads on narrow columns; threads_per_block=256 took 17.3us versus 9.9us
# at 64. Use 32 threads for 1-2 fields and 64 for register-heavy Welford. Against the
# pre-shared kernel this is 0.92-1.01x, except (16384, 1024) sum/amax lose 7-9%,
# while argmax gains 6-8% and wide-short sum gains 15%.
_THREADS_PER_BLOCK = 32
_WIDE_ACC_THREADS_PER_BLOCK = 64  # 3-field traits (Welford): see above
# By default every planned warp (at most 16) gets its own worker; policies that pick
# fewer workers have each stride over several original warp trees.
_ORDERED_WORKER_WARPS = 16
_ORDERED_WORKER_CHOICES = (1, 2, 4, 8, 16)
_ORDERED_TILE_CHOICES = (8, 16, WARP)
# Multi-field lane replays partially unroll their 32-lane loop by this factor. On B200,
# fp32 argmax and bf16 var_mean columns (16, R, 1024) at R = 2048-8192 compiled in
# 0.4-1.2s instead of the static replay's 1.2-12.2s, running at 0.65-1.00x of its speed.
# Single-field replays stay static: rolled, fp32 sums ran at as little as 0.46x.
_ORDERED_LANE_UNROLL = 4


class OrderedColConfig(NamedTuple):
    # Useful 32-column blocks (partials * columns / WARP), whatever tile_columns is.
    min_blocks: int
    worker_warps: int = _ORDERED_WORKER_WARPS
    tile_columns: int = WARP
    full_tiles: bool = False
    # The outputs this mapping was measured with; None applies to any outputs.
    out_dtypes: tuple[torch.dtype, ...] | None = None


_ORDERED_COL_CONFIGS: dict[tuple[int, int], dict[str, OrderedColConfig]] = {}
# _OrderedColReduce would reject these only once a matching view runs.
_bad_mappings = {
    (cc, key): cfg
    for cc, configs in _ORDERED_COL_CONFIGS.items()
    for key, cfg in configs.items()
    if cfg.worker_warps not in _ORDERED_WORKER_CHOICES
    or cfg.tile_columns not in _ORDERED_TILE_CHOICES
}
if _bad_mappings:
    raise ValueError(f"invalid ordered-column mappings: {_bad_mappings}")


def select_ordered_col_config(
    cc: tuple[int, int],
    dtype: torch.dtype,
    trait_key: str,
    columns: int,
    batches: int,
    partials: int,
    *,
    nfields: int,
    acc_bits: int,
    out_dtypes: tuple[torch.dtype, ...],
    rows: int | None = None,
    full_tiles: bool = False,
) -> OrderedColConfig | None:
    """Return the mapping for an ordered column view, or None: looped plans then take the
    default mapping and split plans decline.

    Entries are keyed by trait key. Split plans (partials > 1) without an entry take a
    clamp-free fallback on _ORDERED_COL_CCS, which applies to any outputs. Mappings apply
    to unbatched _ORDERED_COL_DTYPES views with 32-bit accumulators and, if the mapping
    has out_dtypes, only to those outputs. full_tiles means a split plan or an exact
    looped plan; with rows, B200 single-field full-tile views then take measured tile
    widths, and C 128-257 a band of widths. Views that fill fewer than the mapping's
    min_blocks useful 32-column blocks get None.
    """
    cfg = _ORDERED_COL_CONFIGS.get(cc, {}).get(trait_key)
    # Looped exact columns ran within 1% with and without address clamps on B200, so
    # only split plans take the clamp-free fallback.
    if (
        cfg is None
        and cc in _ORDERED_COL_CCS
        and dtype in _ORDERED_COL_DTYPES
        and partials > 1
    ):
        cfg = OrderedColConfig(0, tile_columns=WARP, full_tiles=full_tiles)
    if cfg is None or dtype not in _ORDERED_COL_DTYPES or batches != 1:
        return None
    measured = cfg.out_dtypes
    if acc_bits != 32 or measured is not None and tuple(out_dtypes) != measured:
        return None
    # Measured on routed split views only (one field, B = 1, C >= 128).
    if cc == (10, 0) and full_tiles and rows is not None and nfields == 1:
        if rows == 65536 and columns == 1024:
            cfg = cfg._replace(tile_columns=16)
        # A band of shapes rather than measured anchors.
        elif 32768 <= rows <= 131072 and 128 <= columns <= 257:
            width = WARP if partials >= 16 and columns == 257 else 16
            cfg = cfg._replace(tile_columns=width)
        bf16_amax = dtype == torch.bfloat16 and trait_key == "amax"
        if bf16_amax and rows == 32768 and columns == 4096:
            cfg = cfg._replace(tile_columns=16)
    # Count useful column lanes, excluding padding in the last output tile.
    if partials * columns < WARP * cfg.min_blocks:
        return None
    return cfg


class ColConfig(NamedTuple):
    rule: str
    threads_per_block: int
    npar: int
    vec: int
    partial_layout: str
    combine_columns: int = 0
    unroll: int = tile._ROLL_UNROLL
    kernel_order: Literal["linear", "inner_tree"] = "linear"
    group: int = 1


def select_col_config(
    cc: tuple[int, int],
    dtype: torch.dtype,
    rows: int,
    columns: int,
    batches: int,
    nfields: int,
    trait_key: str,
    *,
    itemsize: int,
    acc_bits: int,
    nouts: int,
    welford_outs: int,
    threads_per_block: int | None = None,
    npar: int | None = None,
    vec: int | None = None,
) -> ColConfig:
    """Select the column kernel, geometry, and partial layout. Any explicit
    threads_per_block, npar or vec overrides the geometry and disables the policy rules.
    welford_outs is traits.welford_outputs(trait).

    Whether a view reaches the column kernel at all is `kernel_general._col_route_ok`'s call.
    """
    explicit = any(v is not None for v in (threads_per_block, npar, vec))
    threads = _WIDE_ACC_THREADS_PER_BLOCK if nfields >= 3 else _THREADS_PER_BLOCK
    # Partials are indexed in Int32 up to outputs * npar.
    outputs = batches * columns
    max_npar = (_INT32_LIMIT - 1) // max(1, outputs)
    if npar is not None and npar > max_npar:
        raise ValueError(f"npar={npar} overflows Int32 partials for {outputs} outputs")
    if npar is not None and npar > _MAX_NPAR:
        raise ValueError(f"npar={npar} exceeds grid y's {_MAX_NPAR} blocks")
    auto_npar, auto_vec = _col_geometry(rows, columns, batches, itemsize, max_npar)
    cfg = ColConfig(
        "explicit" if explicit else "default",
        threads if threads_per_block is None else threads_per_block,
        auto_npar if npar is None else npar,
        auto_vec if vec is None else vec,
        "partition" if batches * columns >= _C_THREAD_STAGE2 else "column",
    )
    row_bytes = columns * itemsize
    size = rows * row_bytes
    # Anchor rows floor-divide the anchor sizes, so a view matches an anchor it falls short
    # of by less than one row.
    anchors = (16 << 20, 64 << 20, 256 << 20, 2 << 30)
    anchor = next((a for a in anchors if a - row_bytes < size <= a), None)
    if (
        not explicit
        and anchor is not None
        and cc == (10, 0)
        and dtype in _B200_DTYPES
        and columns in (257, 4095)
        and acc_bits == 32
        and batches == 1
    ):
        narrow = dtype in (torch.float16, torch.bfloat16)
        welford = welford_outs == nouts
        # Only argmax with int32 indices was measured.
        index = (trait_key, nfields, nouts) == ("argmaxi32", 2, 1)
        one_field = (
            nfields == 1
            and nouts == 1
            and trait_key in ("sum", "prod", "mean", "amax", "amin")
        )
        if columns == 4095 and anchor == 16 << 20 and (one_field or welford):
            return cfg._replace(
                rule="b200_ragged_c4095_ordered",
                kernel_order="inner_tree",
            )
        if columns == 257 and welford:
            npars = {
                16 << 20: 512,
                64 << 20: 512,
                256 << 20: 1024 if narrow else 512,
                2 << 30: 4096,
            }
            return cfg._replace(
                rule="b200_ragged_welford_c257",
                threads_per_block=64,
                npar=npars[anchor],
                vec=1,
                group=16 if narrow else 8,
            )
        if columns == 4095 and welford:
            return cfg._replace(
                rule="b200_ragged_welford_c4095",
                threads_per_block=64,
                npar=1928 if anchor == 2 << 30 else 128,
                vec=1,
                group=8,
            )
        if columns == 257 and (one_field or index):
            npars = {
                16 << 20: 1024,
                64 << 20: 2048,
                256 << 20: 4096,
                2 << 30: 16384 if narrow else 8192,
            }
            return cfg._replace(
                rule="b200_ragged_c257",
                threads_per_block=64,
                npar=npars[anchor],
                vec=1,
            )
        if columns == 4095 and one_field:
            cfg = cfg._replace(
                rule="b200_ragged_c4095",
                threads_per_block=64,
                npar=512 if anchor == 2 << 30 else 256,
                vec=1,
            )
            if narrow and anchor == 256 << 20:
                return cfg._replace(
                    npar=512, partial_layout="partition", combine_columns=8, unroll=16
                )
            return cfg
        if columns == 4095 and index and anchor == 16 << 20 and not narrow:
            return cfg._replace(
                rule="b200_ragged_argmax_c4095_small",
                threads_per_block=32,
                npar=8,
                vec=1,
                unroll=64,
            )
        if columns == 4095 and index and anchor >= 256 << 20:
            npars = {
                256 << 20: 512 if narrow else 128,
                2 << 30: 1024 if narrow else 512,
            }
            return cfg._replace(
                rule="b200_ragged_argmax_c4095",
                threads_per_block=64,
                npar=npars[anchor],
                vec=1,
            )
    return cfg


def _col_geometry(
    rows: int, columns: int, batches: int, itemsize: int, max_npar: int
) -> tuple[int, int]:
    """Default (npar, vec) for stage one."""
    vec = tile.vec_size(columns, itemsize)
    while vec > 1 and batches * columns // vec < _MIN_LANES:
        vec //= 2
    if rows <= _SHORT_R:
        return 1, min(vec, 2)
    while True:
        lanes = batches * columns // vec
        npar = min(rows // _Q_MIN, -(-_TARGET_LANES // lanes), max_npar, _MAX_NPAR)
        npar = max(1, npar)
        if rows // npar >= _Q_VEC or vec == 1:
            return npar, vec
        vec //= 2


class PartialColumns:
    """Combine partition-major states while coalescing adjacent column loads.

    TileReduce's combine mode folds each output's partitions serially in one thread. This
    kernel strides thread groups over partitions and merges them in shared memory, a second
    schedule that would otherwise add smem and branching to TileReduce's shared body.
    """

    def __init__(self, trait, tile_columns, nouts):
        # The tuned widths; each tiles the 128-thread block exactly.
        if tile_columns not in (8, 16, 32):
            raise ValueError(f"combine_columns must be 8, 16 or 32, got {tile_columns}")
        self.trait = trait
        self.tile_columns = tile_columns
        self.threads = 128
        self.groups = self.threads // tile_columns
        self.nouts = nouts

    @property
    def cache_sig(self):
        return (self.tile_columns, self.nouts)

    @cute.jit
    def __call__(
        self,
        ins: list,
        outs: list,
        columns: Int32,
        rows: Int32,
        partitions: Int32,
        stream,
    ):
        self.kernel(ins, outs, columns, rows, partitions).launch(
            grid=[cute.ceil_div(columns, self.tile_columns), 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, ins: list, outs: list, columns: Int32, rows: Int32, partitions: Int32
    ):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        lane = Int32(tx) % Int32(self.tile_columns)
        group = Int32(tx) // Int32(self.tile_columns)
        col = Int32(bx) * Int32(self.tile_columns) + lane
        safe_col = col if col < columns else Int32(0)
        trait = self.trait
        combine, fdtypes = trait.combine, trait.fdtypes
        nf = const_expr(trait.nfields)
        identity = trait.init()
        acc = identity
        for p in cutlass.range(group, partitions, self.groups):
            # select_col_config caps outputs * npar below 2^31.
            offset = p * columns + safe_col
            acc = combine(acc, tuple(fdtypes[f](ins[f][offset]) for f in range(nf)))
        smem = cutlass.utils.SmemAllocator()
        partials = [
            smem.allocate_tensor(
                fdtypes[f], cute.make_layout(self.threads), byte_alignment=8
            )
            for f in range(nf)
        ]
        for f in cutlass.range_constexpr(nf):
            partials[f][tx] = acc[f]
        cute.arch.barrier()
        if group == Int32(0):
            acc = identity
            for g in cutlass.range_constexpr(self.groups):
                offset = Int32(g * self.tile_columns) + lane
                acc = combine(
                    acc, tuple(fdtypes[f](partials[f][offset]) for f in range(nf))
                )
        # Keep the Python trait out of runtime branch state.
        result = trait.project(acc, trait.acc(rows))
        if (group == Int32(0)) & (col < columns):
            if const_expr(self.nouts == 1):
                outs[0][col] = outs[0].element_type(result)
            else:
                for f in cutlass.range_constexpr(self.nouts):
                    outs[f][col] = outs[f].element_type(result[f])


class _OrderedColReduce:
    """Preserve the row inner-tree DAG while coalescing adjacent column loads."""

    def __init__(
        self,
        trait: Any,
        plan: Any,
        R: int,
        C: int,
        nouts: int,
        *,
        worker_warps: int = _ORDERED_WORKER_WARPS,
        tile_columns: int = WARP,
        full_tiles: bool = False,
        partition_begin: int = 0,
        partition_count: int | None = None,
    ) -> None:
        if plan.shape not in ("looped", "split"):
            raise ValueError(f"ordered columns do not support {plan.shape=}")
        if plan.vec_linear:
            raise ValueError("ordered columns replay tree vector folds only")
        if len(plan.batches) != 1:
            raise ValueError("ordered columns fold a single planner batch")
        if worker_warps not in _ORDERED_WORKER_CHOICES:
            raise ValueError(
                f"worker_warps must be one of {_ORDERED_WORKER_CHOICES}, got {worker_warps}"
            )
        if tile_columns not in _ORDERED_TILE_CHOICES:
            raise ValueError(
                f"tile_columns must be one of {_ORDERED_TILE_CHOICES}, got {tile_columns}"
            )
        self.trait = trait
        self.plan = plan
        self.R = R
        # C and the batch count are runtime, so each R compiles once. Each leaf costs an
        # address multiply-add that a compile-time C folds into the load, but the clamp
        # keeps addresses independent: against per-C kernels, B200 sum, argmax and
        # var_mean (fp32/bf16) ran at 1.04x geomean (0.79-1.70x) for looped R 128-8192
        # and 1.18x (0.69-1.68x) for split R 32768-131072, and a new C skips a 0.8 s
        # (median) compile. Full tiles read rows in order, where per-C kernels fold the
        # offsets into loads: there runtime C ran at 0.93x looped and 0.97x split
        # (best tile mode each; short looped argmax 0.68-0.79x). Row views take a
        # 64-bit stride once R * C needs one.
        self.wide = R * C >= _INT32_LIMIT
        self.nouts = nouts
        self.partition_begin = partition_begin
        partitions = plan.split[0] if plan.shape == "split" else 1
        if partition_count is None:
            partition_count = partitions - partition_begin
        self.partition_count = partition_count
        end = partition_begin + partition_count
        split_end = plan.shape == "split" and end > partitions
        if partition_begin < 0 or partition_count < 1 or split_end:
            raise ValueError(
                f"ordered-column partition range is out of bounds: {partition_begin=}, "
                f"{partition_count=} for {partitions} partitions"
            )
        if plan.shape != "split" and (partition_begin, self.partition_count) != (0, 1):
            raise ValueError("only split plans have ordered-column partitions")
        # Workers stride over the plan's warp trees; rows with fewer warps use one each.
        self.workers = min(worker_warps, plan.wpr)
        self.tile_columns = tile_columns
        # A ragged final split partition still needs address clamps.
        ragged_tail = (
            plan.shape == "split"
            and plan.split[1] != plan.split[2]
            and self.partition_begin + self.partition_count == plan.split[0]
        )
        # Exact looped plans tile the row without padded warps or leaves past R. Skipping
        # their leaf masks, and the shared-memory merge for one-warp rows, ran B200 argmax
        # at up to 1.4x and var_mean at up to 2.4x on exact views; sums were within noise.
        self.exact = plan.shape == "looped" and plan.tms[0].exact
        # Split warps tile each plan.split[1]-row partition exactly.
        whole = plan.shape == "split" or self.exact
        self.full_tiles = full_tiles and whole and not ragged_tail

    @property
    def cache_sig(self) -> tuple[Any, ...]:
        key = (
            self.R,
            self.wide,
            self.trait.nfields,
            self.nouts,
            self.workers,
            self.tile_columns,
            self.full_tiles,
            self.partition_begin,
            self.partition_count,
        )
        return key + (self.plan.sig,)

    @cute.jit
    def __call__(
        self,
        mIn: cute.Tensor,
        mOuts: list,
        in_base: Int64,
        num_o: Int32,
        C: Int32,
        stream,
    ):
        self.kernel(mIn, mOuts, in_base, num_o, C).launch(
            grid=[cute.ceil_div(num_o, self.tile_columns), self.partition_count, 1],
            block=[self.tile_columns, self.workers, 1],
            stream=stream,
        )

    @cute.jit
    def _leaf(self, rows, warp_base, k, active, bound, last):
        trait = self.trait
        red = warp_base + k
        if const_expr(self.full_tiles):
            # Every leaf is in bounds, and inactive lanes never store their outputs.
            return trait.leaf(rows[k], red)
        if const_expr(self.exact):
            # As above, but the clamp keeps each address independent of the last one.
            return trait.leaf(rows[cutlass.min(Int32(k), last)], red)
        valid = active & (red < bound)
        # Leaves past the bound read its last in-bounds row; the select below discards
        # them. The clamp also keeps each address independent of the last one.
        got = trait.leaf(rows[cutlass.min(Int32(k), last)], red if valid else Int32(0))
        ident = trait.init()
        return tuple(got[f] if valid else ident[f] for f in range(trait.nfields))

    @cute.jit
    def _lane_root(self, rows, warp_base, k, active, bound, last):
        vals = [
            self._leaf(rows, warp_base, k + i, active, bound, last)
            for i in range(self.plan.vec)
        ]
        return tile._reduce_vec(vals, self.plan.vec, self.trait.combine)

    @cute.jit
    def _warp_root(self, mIn, col_base, C, split_base, partition, active, warp):
        """Replay one original warp's tree over its share of the rows."""
        trait, plan = self.trait, self.plan
        op, nf = trait.combine, const_expr(trait.nfields)
        loads, warp_chunk = plan.batches[0][2:]
        warp_base = split_base + warp * Int32(warp_chunk)
        bound = Int32(self.R)
        if const_expr(plan.shape == "split" and plan.split[1] != plan.split[2]):
            nbatch, bte, tail, full_chunk, last_chunk = plan.split
            final_batch = partition == Int32(nbatch - 1)
            remaining = Int32(tail) if final_batch else Int32(bte)
            chunk = Int32(last_chunk) if final_batch else Int32(full_chunk)
            offset = warp * chunk
            offset = remaining if offset > remaining else offset  # noqa: FURB136
            warp_base = split_base + offset
            end = offset + chunk
            end = remaining if end > remaining else end  # noqa: FURB136
            bound = split_base + end
        # A padded or ragged warp can start at its bound; its leaves all mask, so its
        # view starts at most at R - 1.
        first = warp_base if warp_base < Int32(self.R) else Int32(self.R - 1)
        rows = cute.make_tensor(
            mIn.iterator + col_base + Int64(first) * Int64(C),
            cute.make_layout(
                const_expr(loads * WARP * plan.vec),
                stride=Int64(C) if const_expr(self.wide) else C,
            ),
        )
        last = cutlass.max(bound - Int32(1) - first, Int32(0))
        load_tree = []
        for load in cutlass.range_constexpr(loads):
            if const_expr(nf == 1):
                lane_roots = cute.make_rmem_tensor(
                    cute.make_layout(WARP), trait.fdtypes[0]
                )
                for original_lane in cutlass.range_constexpr(WARP):
                    lane_roots[original_lane] = self._lane_root(
                        rows,
                        warp_base,
                        (load * WARP + original_lane) * plan.vec,
                        active,
                        bound,
                        last,
                    )[0]
                load_root = tile._inner_tree_reduce(
                    [(lane_roots[i],) for i in range(WARP)], op
                )
            else:
                lane_roots = [
                    cute.make_rmem_tensor(cute.make_layout(WARP), trait.fdtypes[f])
                    for f in range(nf)
                ]
                for original_lane in cutlass.range(WARP, unroll=_ORDERED_LANE_UNROLL):
                    root = self._lane_root(
                        rows,
                        warp_base,
                        Int32(load * WARP * plan.vec)
                        + Int32(original_lane) * Int32(plan.vec),
                        active,
                        bound,
                        last,
                    )
                    for f in cutlass.range_constexpr(nf):
                        lane_roots[f][original_lane] = root[f]
                load_root = tile._inner_tree_reduce(
                    [tuple(lane_roots[f][i] for f in range(nf)) for i in range(WARP)],
                    op,
                )
            tile._streaming_push(load_tree, load_root, load, plan.depth, op)
        return load_tree[0]

    @cute.kernel
    def kernel(
        self, mIn: cute.Tensor, mOuts: list, in_base: Int64, num_o: Int32, C: Int32
    ):
        tx, ty, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        trait, plan = self.trait, self.plan
        op, ident = trait.combine, trait.init()
        split_base = Int32(0)
        partition = Int32(by) + Int32(const_expr(self.partition_begin))
        if const_expr(plan.shape == "split"):
            split_base = partition * Int32(plan.split[1])

        lane, worker = Int32(tx), Int32(ty)
        columns = const_expr(self.tile_columns)
        output = Int32(bx) * Int32(columns) + lane
        active = output < num_o
        safe = output if active else Int32(0)
        batch, col = safe // C, safe % C
        col_base = in_base + Int64(batch) * Int64(self.R) * Int64(C) + Int64(col)
        nf = const_expr(trait.nfields)
        if const_expr(plan.wpr == 1):
            # One warp replays the whole row, so its root needs no shared-memory merge.
            # Only looped plans run one warp; like the indexed tree, seed the identity
            # so -0 sums become +0.
            root = self._warp_root(
                mIn, col_base, C, split_base, partition, active, Int32(0)
            )
            final = op(ident, root)
        else:
            smem = cutlass.utils.SmemAllocator()
            roots = [
                smem.allocate_tensor(
                    trait.fdtypes[f],
                    cute.make_layout(const_expr(plan.wpr * columns)),
                    byte_alignment=8,
                )
                for f in range(nf)
            ]
            if const_expr(self.workers == plan.wpr):
                # One original warp per worker needs no loop.
                root = self._warp_root(
                    mIn, col_base, C, split_base, partition, active, worker
                )
                for f in cutlass.range_constexpr(nf):
                    roots[f][worker * Int32(columns) + lane] = root[f]
            else:
                for warp in cutlass.range(worker, plan.wpr, self.workers):
                    warp_root = self._warp_root(
                        mIn, col_base, C, split_base, partition, active, warp
                    )
                    for f in cutlass.range_constexpr(nf):
                        roots[f][warp * Int32(columns) + lane] = warp_root[f]
            cute.arch.barrier()
            groups = const_expr(plan.wpr // plan.kchunk)
            final = ident
            if worker == Int32(0):
                group_vals = [
                    tile._inner_tree_reduce(
                        [
                            tuple(
                                roots[f][(g * plan.kchunk + i) * Int32(columns) + lane]
                                for f in range(nf)
                            )
                            for i in range(plan.kchunk)
                        ],
                        op,
                    )
                    for g in range(groups)
                ]
                # Like TileReduce._fold_itree, only several groups merge across 32 lanes.
                merged = group_vals[0]
                if const_expr(groups > 1):
                    padding = ident
                    # Constant identities fold Welford's 0 / 0 count ratio at trace time.
                    if const_expr(nf > 1):
                        dynamic_false = lane < Int32(0)
                        padding = tuple(
                            group_vals[0][f] if dynamic_false else ident[f]
                            for f in range(nf)
                        )
                    group_vals.extend([padding] * (WARP - groups))
                    merged = tile._inner_tree_reduce(group_vals, op)
                if const_expr(plan.shape == "split"):
                    final = merged
                else:
                    # The indexed tree seeds its fold with the identity, so -0 sums become +0.
                    final = op(ident, merged)

        if const_expr(plan.shape == "split"):
            if (worker == Int32(0)) & active:
                index = output * Int32(plan.split[0]) + partition
                for f in cutlass.range_constexpr(trait.nfields):
                    mOuts[f][index] = final[f]
        else:
            projected = trait.project(final, trait.acc(self.R))
            if (worker == Int32(0)) & active:
                if const_expr(self.nouts == 1):
                    mOuts[0][output] = mOuts[0].element_type(projected)
                else:
                    for k in cutlass.range_constexpr(self.nouts):
                        mOuts[k][output] = mOuts[k].element_type(projected[k])


def reduce_ordered_col(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtypes: Sequence[torch.dtype],
    nouts: int,
    *,
    cc: tuple[int, int] | None = None,
    config: OrderedColConfig | None = None,
) -> tuple[torch.Tensor, ...] | None:
    """Reduce axis -2 of a contiguous (R, C) or (B, R, C) view, or return None.

    An explicit config is used as-is; otherwise select_ordered_col_config picks one for
    compute capability cc, x's device's by default. Split plans run only with a config,
    explicit or selected; looped plans default to OrderedColConfig(0).
    """
    if not x.is_cuda:
        raise AssertionError(f"want a CUDA input, got {x.device}")
    device = x.get_device()
    check_current(device)
    if x.dim() == 2 and x.is_contiguous():
        B, (R, C) = 1, x.shape
        out_shape = (C,)
    elif x.dim() == 3 and x.is_contiguous():
        B, R, C = x.shape
        out_shape = (B, C)
    else:
        return None
    if len(out_dtypes) < nouts:
        raise AssertionError(f"want {nouts} output dtypes, got {len(out_dtypes)}")
    # The last block's Int32 output index must not wrap past the outputs.
    last = B * C + WARP
    if C < WARP or last >= _INT32_LIMIT:
        return None
    plan = trait_itree_plan(
        trait, R, B * C, x.element_size(), stage=False, device=x.device
    )
    # The planner builds no linear vector folds here; declining one guards future planner
    # changes, since this kernel replays tree folds only.
    if plan is None or plan.shape not in ("looped", "split") or plan.vec_linear:
        return None
    # Single-field traits replay all 32 lanes' leaves statically. Multi-field traits roll
    # lanes but replay a static lane tree per load, which compiles about as slowly (1.3 s
    # against 1.5 s at fp32 R = 8192). Cap both so long looped rows keep the indexed
    # tree's compile time. Plans past one planner batch always exceed the cap.
    if sum(b[2] for b in plan.batches) * WARP * plan.vec > tile.MAX_UNROLL:
        return None
    split = plan.shape == "split"
    # Split is verified bitwise for floating inputs only. Partials are indexed in Int32
    # and partitions ride grid y.
    floating = (torch.float32, torch.bfloat16, torch.float16, torch.float64)
    if split and (
        x.dtype not in floating
        or B * C * plan.split[0] >= _INT32_LIMIT
        or plan.split[0] > _MAX_NPAR
    ):
        return None
    if config is None:
        config = select_ordered_col_config(
            _hw.caps(x.device).cc if cc is None else cc,
            x.dtype,
            trait_key,
            C,
            B,
            plan.split[0] if split else 1,
            nfields=trait.nfields,
            acc_bits=max(dt.width for dt in trait.fdtypes),
            out_dtypes=tuple(out_dtypes[:nouts]),
            rows=R,
            full_tiles=split or plan.tms[0].exact,
        )
        if config is None:
            if split:
                return None
            config = OrderedColConfig(0)
    workers, columns = config.worker_warps, config.tile_columns
    # Only full tiles split off the ragged final partition, which keeps its clamps.
    ragged = split and plan.split[1] != plan.split[2]
    if ragged and config.full_tiles:
        ops = (
            _OrderedColReduce(
                trait,
                plan,
                R,
                C,
                trait.nfields,
                full_tiles=True,
                partition_count=plan.split[0] - 1,
                worker_warps=workers,
                tile_columns=columns,
            ),
            _OrderedColReduce(
                trait,
                plan,
                R,
                C,
                trait.nfields,
                partition_begin=plan.split[0] - 1,
                partition_count=1,
                worker_warps=workers,
                tile_columns=columns,
            ),
        )
    else:
        ops = (
            _OrderedColReduce(
                trait,
                plan,
                R,
                C,
                trait.nfields if split else nouts,
                full_tiles=config.full_tiles,
                worker_warps=workers,
                tile_columns=columns,
            ),
        )
    outs = [
        torch.empty(out_shape, device=x.device, dtype=dtype)
        for dtype in out_dtypes[:nouts]
    ]
    kernel_x = _flat(x)
    kernel_outs = (
        [
            torch.empty(B * C * plan.split[0], device=x.device, dtype=cute2torch[d])
            for d in trait.fdtypes
        ]
        if split
        else [out.reshape(-1) for out in outs]
    )

    def _fake():
        fake_in = _L.fake_compact(torch2cute[kernel_x.dtype], (cute.sym_int64(),))
        fake_outs = [
            _L.fake_compact(torch2cute[out.dtype], (_L.sym(),)) for out in kernel_outs
        ]
        return fake_in, fake_outs

    args = (Int64(x.storage_offset()), Int32(B * C), Int32(C), _stream(device))
    for op in ops:
        key = (
            "ordered_col1" if split else "ordered_col",
            trait_key,
            x.dtype,
            tuple(out.dtype for out in kernel_outs),
            device,
        ) + op.cache_sig
        build = lambda: _compile(op, *_fake(), *args)  # noqa: E731
        cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")(
            _L.read_only(kernel_x), kernel_outs, *args
        )
    if split:
        flat_outs = [out.reshape(-1) for out in outs]
        _combine_itree_partials(
            trait, trait_key, plan, kernel_outs, flat_outs, R, nouts
        )
    return tuple(outs)


def reduce_col_tile(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtype: torch.dtype,
    threads_per_block: int | None = None,
    npar: int | None = None,
    vec: int | None = None,
) -> torch.Tensor:
    """Reduce dim 0 of contiguous 2D x to (C,), splitting it npar ways."""
    if x.dim() != 2:
        raise AssertionError(f"want 2D x, got {tuple(x.shape)}")
    return _reduce_col_tile(
        trait,
        trait_key,
        x,
        [out_dtype],
        1,
        threads_per_block,
        npar,
        vec,
    )[0]


def reduce_col_tile_2out(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtypes: Sequence[torch.dtype],
    threads_per_block: int | None = None,
    npar: int | None = None,
    vec: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Reduce dim 0 of contiguous 2D x to two outputs per column, like reduce_col_tile."""
    if x.dim() != 2:
        raise AssertionError(f"want 2D x, got {tuple(x.shape)}")
    return _reduce_col_tile(
        trait,
        trait_key,
        x,
        out_dtypes,
        2,
        threads_per_block,
        npar,
        vec,
    )


def reduce_batched_col_tile(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtypes: Sequence[torch.dtype],
    nouts: int,
    threads_per_block: int | None = None,
    npar: int | None = None,
    vec: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Reduce dim 1 of contiguous 3D x to nouts (B, C) outputs."""
    if x.dim() != 3:
        raise AssertionError(f"want 3D x, got {tuple(x.shape)}")
    return _reduce_col_tile(
        trait, trait_key, x, out_dtypes, nouts, threads_per_block, npar, vec
    )


def _reduce_col_tile(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtypes: Sequence[torch.dtype],
    nouts: int,
    threads_per_block: int | None,
    npar: int | None,
    vec: int | None,
) -> tuple[torch.Tensor, ...]:
    # Stage one compiles a packed row-major descriptor.
    if x.dim() not in (2, 3) or not x.is_cuda or not x.is_contiguous():
        raise AssertionError(f"want contiguous 2D/3D CUDA, got {tuple(x.shape)}")
    device = x.get_device()
    check_current(device)
    if len(out_dtypes) < nouts:
        raise AssertionError(f"want {nouts} output dtypes, got {len(out_dtypes)}")
    if x.dim() == 2:
        B, (R, C) = 1, x.shape
        rows = x
        out_shape = (C,)
    else:
        B, R, C = x.shape
        # Stage one indexes batched rows as batch * R + row in Int32.
        if B * R >= _INT32_LIMIT:
            raise AssertionError(f"want B * R < 2^31, got B={B}, R={R}")
        rows = x.reshape(B * R, C)
        out_shape = (B, C)
    if B * C >= _INT32_LIMIT:
        raise AssertionError(f"want B * C < 2^31 outputs, got B={B}, C={C}")
    align = _L.supported_alignment(rows, tile.align_bytes(C, x.element_size()))
    cc = _hw.caps(x.device).cc
    cfg = select_col_config(
        cc,
        x.dtype,
        R,
        C,
        B,
        trait.nfields,
        trait_key,
        itemsize=x.element_size(),
        acc_bits=max(dt.width for dt in trait.fdtypes),
        nouts=nouts,
        welford_outs=welford_outputs(trait),
        threads_per_block=threads_per_block,
        npar=npar,
        vec=vec,
    )
    if cfg.kernel_order not in ("linear", "inner_tree"):
        raise ValueError(f"unknown kernel order {cfg.kernel_order!r}")
    if cfg.kernel_order == "inner_tree":
        ordered = reduce_ordered_col(trait, trait_key, x, out_dtypes, nouts, cc=cc)
        if ordered is not None:
            return ordered
    threads_per_block, npar, vec = cfg.threads_per_block, cfg.npar, cfg.vec
    if vec <= 0:
        raise ValueError(f"vec must be positive, got {vec}")
    if C % vec:
        # An explicit nondivisor vec would leave trailing outputs uninitialized.
        raise AssertionError(f"vec must divide the column count: {C=} {vec=}")
    if npar <= 0:
        raise ValueError(f"npar must be positive, got {npar}")
    outs = [
        torch.empty(out_shape, device=x.device, dtype=dtype)
        for dtype in out_dtypes[:nouts]
    ]
    kernel_x = rows
    kernel_outs = [out.reshape(-1) for out in outs]
    total = B * C
    nchunks = Int32(total // vec)
    batch_chunks = Int32(C // vec) if x.dim() == 3 else None
    q, nrows = Int32(-(-R // npar)), Int32(R)

    single = npar == 1
    if cfg.partial_layout not in ("partition", "column"):
        raise ValueError(f"unknown partial layout {cfg.partial_layout!r}")
    pc = cfg.partial_layout == "partition"
    width = cfg.combine_columns
    if width and not pc:
        raise ValueError("combine_columns reads partition-major partials")
    # Build the tiled combine first so an invalid width raises before stage one launches.
    tiled = PartialColumns(trait, width, nouts) if width else None
    op = tile.TileReduce(
        trait,
        torch2cute[kernel_x.dtype],
        "col",
        C,
        threads_per_block=threads_per_block,
        nouts=nouts,
        final=single,
        vec=vec,
        batched_col=x.dim() == 3,
        pc=pc,
        unroll=cfg.unroll,
        col_group=cfg.group,
    )
    parts = (
        []
        if single
        else [
            torch.empty(
                total * npar,
                device=x.device,
                dtype=cute2torch[trait.fdtypes[f]],
            )
            for f in range(trait.nfields)
        ]
    )
    dsts = outs if single else parts
    kernel_dsts = [dst.reshape(-1) for dst in dsts]

    def _fake():
        # Dynamic descriptors require vec divisibility; None avoids a 1.27x unused argument.
        return (
            [
                _L.fake_compact(
                    torch2cute[kernel_x.dtype],
                    (_L.sym(), _L.sym(vec)),
                    stride_order=(1, 0),
                    align=align,
                )
            ],
            [_L.fake_compact(torch2cute[d.dtype], (_L.sym(),)) for d in kernel_dsts],
            nchunks,
            batch_chunks,  # nwaves slot: column chunks per batch
            nrows,
            q,
            Int32(npar),
            None,  # the general axis's decode: exts, strides, in_base, limit
            None,
            None,
            None,
            None,
            None,
            _stream(device),
        )

    # vec can be narrower than the compile-time alignment, so key both.
    key = (
        "coltile",
        trait_key,
        x.dtype,
        tuple(out_dtypes[:nouts]),
        align,
        device,
    ) + op.cache_sig
    build = lambda: _compile(op, *_fake())  # noqa: E731
    cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")(
        [_L.read_only(kernel_x)],
        kernel_dsts,
        nchunks,
        batch_chunks,
        nrows,
        q,
        Int32(npar),
        None,
        None,
        None,
        None,
        None,
        None,
        _stream(device),
    )
    if single:
        return tuple(outs)

    if tiled is not None:
        pdt = tuple(pp.dtype for pp in parts)
        key2 = (
            "coltile2t",
            trait_key,
            tuple(out_dtypes[:nouts]),
            pdt,
            device,
        ) + tiled.cache_sig
        build2 = lambda: _compile(  # noqa: E731
            tiled,
            [_L.fake_compact(torch2cute[pp.dtype], (_L.sym(),)) for pp in parts],
            [
                _L.fake_compact(torch2cute[out.dtype], (_L.sym(),))
                for out in kernel_outs
            ],
            Int32(total),
            nrows,
            Int32(npar),
            _stream(device),
        )
        ins = [_L.read_only(pp) for pp in parts]
        cached_plan(_CACHE, key2, build2, op=f"aten::{trait_key}")(
            ins, kernel_outs, Int32(total), nrows, Int32(npar), _stream(device)
        )
        return tuple(outs)

    # Fold npar partials and project with true R; partition-major partials use threads per column.
    if not pc:
        s2 = ReduceBlock(
            trait,
            count=npar,
            num_o=total,
            red_pairs=[(npar, 1)],
            kept_pairs=[(total, npar)],
            from_partials=True,
            project_n=R,
            nouts=nouts,
            final=True,
            block=128,
        )
        pdt = tuple(pp.dtype for pp in parts)
        key2 = (
            "coltile2b",
            trait_key,
            tuple(out_dtypes[:nouts]),
            pdt,
        ) + s2.cache_sig
        _launch(s2, key2, parts, kernel_outs)
        return tuple(outs)

    # Shared combine mode uses nchunks as C and nrows as true R; q is unused.
    op2 = tile.TileReduce(
        trait,
        torch2cute[kernel_x.dtype],
        "col",
        total,
        threads_per_block=threads_per_block,
        nouts=nouts,
        vec=1,
        combine=True,
    )

    def _fake2():
        # Row nwaves and split q are unused.
        return (
            [_L.fake_compact(torch2cute[pp.dtype], (_L.sym(),)) for pp in parts],
            [
                _L.fake_compact(torch2cute[out.dtype], (_L.sym(),))
                for out in kernel_outs
            ],
            Int32(total),
            None,  # nwaves
            Int32(R),
            None,  # q
            Int32(npar),
            None,  # the general axis's decode
            None,
            None,
            None,
            None,
            None,
            _stream(device),
        )

    pdt = tuple(pp.dtype for pp in parts)
    key2 = (
        "coltile2",
        trait_key,
        x.dtype,
        tuple(out_dtypes[:nouts]),
        pdt,
        device,
    ) + op2.cache_sig
    build2 = lambda: _compile(op2, *_fake2())  # noqa: E731
    cached_plan(_CACHE, key2, build2, op=f"aten::{trait_key}")(
        [_L.read_only(pp) for pp in parts],
        kernel_outs,
        Int32(total),
        None,
        Int32(R),
        None,
        Int32(npar),
        None,
        None,
        None,
        None,
        None,
        None,
        _stream(device),
    )
    return tuple(outs)
