# Column-reduction driver for tile.TileReduce, with launch policy and plan cache, plus the
# PartialColumns stage-two kernel.
# Threads own vectorized kept-axis outputs without lane merging. Splitting the reduced
# axis provides parallelism; unsplit (65536, 256) took 7830us versus ATen's 15.8us.

from collections.abc import Sequence
from typing import Any, Literal, NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64

import torch

from ...cutedsl import hw_caps as _hw, launch as _L
from ...cutedsl.dtypes import cute2torch, torch2cute
from ...cutedsl.plan_cache import cached_plan
from . import tile
from .kernel_general import (
    _combine_itree_partials,
    _flat,
    _INT32_LIMIT,
    _launch,
    ReduceBlock,
)
from .kernel_rowtile import trait_itree_plan
from .traits import WARP, welford_nouts


_compile = _L.compile_kernel
_stream = _L.stream
_CACHE = {}

# About 64 reduced rows per chunk was consistently optimal; fixed P cut tall-narrow
# throughput to one quarter.
_Q_TARGET = 64
_P_MAX = 4096
# Use blocks per column below the measured 4096-16384 crossover, then threads per column.
_C_THREAD_STAGE2 = 8192
# vec controls both load width and live accumulators; cap at 4 because bf16 vec 8 is 0.77-0.83x.
_VEC_MAX = 4
# Small blocks avoid idle threads on narrow columns; threads_per_block=256 took 17.3us versus 9.9us
# at 64. Use 32 threads for 1-2 fields and 64 for register-heavy Welford. Against the
# pre-shared kernel this is 0.92-1.01x, except (16384, 1024) sum/amax lose 7-9%,
# while argmax gains 6-8% and wide-short sum gains 15%.
_THREADS_PER_BLOCK = 32
_WIDE_ACC_THREADS_PER_BLOCK = 64  # 3-field traits (Welford): see above
_ORDERED_WORKER_WARPS = 16
# Multi-field lane replays roll by this factor. On B200, ordered argmax and var_mean
# columns compiled in 0.8-1.2s instead of 7.8-8.6s and ran at 0.92-1.06x. Single-field
# replays stay static: rolled, fp32 sums ran at as little as 0.46x.
_ORDERED_LANE_UNROLL = 4


class OrderedColConfig(NamedTuple):
    min_blocks: int
    worker_warps: int = _ORDERED_WORKER_WARPS
    tile_columns: int = WARP
    full_tiles: bool = False


_ORDERED_COL_CONFIGS: dict[tuple[int, int], dict[str, OrderedColConfig]] = {}


def select_ordered_col_config(
    cc: tuple[int, int],
    dtype: torch.dtype,
    trait_key: str,
    columns: int,
    batches: int,
    partials: int,
    *,
    field_bits: tuple[int, ...],
    out_dtypes: tuple[torch.dtype, ...],
    rows: int | None = None,
    full_tiles: bool = False,
) -> OrderedColConfig | None:
    configs = _ORDERED_COL_CONFIGS.get(cc, {})
    welford = welford_nouts(trait_key)
    cfg = configs.get(trait_key)
    if cfg is None and welford:
        cfg = configs.get("varmean0" if welford == 2 else "var0")
    if (
        cfg is None
        and cc == (10, 0)
        and dtype in (torch.float16, torch.bfloat16, torch.float32)
        and (partials > 1 or full_tiles)
    ):
        cfg = OrderedColConfig(0, tile_columns=WARP, full_tiles=True)
    if (
        cfg is None
        or dtype not in (torch.float32, torch.bfloat16, torch.float16)
        or batches != 1
        or columns < WARP
        or partials < 1
    ):
        return None
    fields = 3 if welford else 2 if trait_key == "argmaxi32" else 1
    output = (
        dtype
        if trait_key in ("amax", "amin") or fields == 3
        else torch.int64
        if fields == 2
        else torch.float32
    )
    nouts = max(welford, 1)
    # B200 single-field reductions may also store in the input dtype, as overrides request.
    narrow = cc == (10, 0) and fields == 1 and out_dtypes == (dtype,)
    if field_bits != (32,) * fields or out_dtypes != (output,) * nouts and not narrow:
        return None
    # Only routed split views (one field, B = 1, C >= 128) reach these rows.
    if cc == (10, 0) and full_tiles and rows is not None:
        if rows == 65536 and columns == 1024:
            cfg = cfg._replace(tile_columns=16, full_tiles=True)
        elif 32768 <= rows <= 131072 and 128 <= columns <= 257:
            width = WARP if partials >= 16 and columns == 257 else 16
            cfg = OrderedColConfig(0, tile_columns=width, full_tiles=True)
        bf16_amax = dtype == torch.bfloat16 and trait_key == "amax"
        if bf16_amax and rows == 32768 and columns == 4096:
            cfg = cfg._replace(tile_columns=16)
    # Count useful column lanes, excluding padding in the last output tile.
    if partials * columns < WARP * cfg.min_blocks:
        return None
    return cfg


def _full_column_tiles(plan: Any, rows: int) -> bool:
    if not plan.batches:
        return False
    limit = plan.split[1] if plan.shape == "split" else rows
    return all(
        off + (plan.wpr - 1) * chunk + loads * WARP * plan.vec <= limit
        for off, _, loads, chunk in plan.batches
    )


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
    alignment: int,
    contiguous: bool,
    threads_per_block: int | None = None,
    npar: int | None = None,
    vec: int | None = None,
) -> ColConfig:
    """Select the column kernel, geometry, and partial layout; explicit arguments override the geometry.

    Whether a view reaches the column kernel at all is `kernel_general._col_route_ok`'s call.
    """
    explicit = any(v is not None for v in (threads_per_block, npar, vec))
    threads = _WIDE_ACC_THREADS_PER_BLOCK if nfields >= 3 else _THREADS_PER_BLOCK
    # Partials are indexed in Int32 up to outputs * npar.
    outputs = batches * columns
    max_npar = (_INT32_LIMIT - 1) // max(1, outputs)
    if npar is not None and npar > max_npar:
        raise ValueError(f"npar={npar} overflows Int32 partials for {outputs} outputs")
    cfg = ColConfig(
        "explicit" if explicit else "default",
        threads if threads_per_block is None else threads_per_block,
        min(_split_p(rows), max_npar) if npar is None else npar,
        min(tile.vec_size(columns, itemsize), _VEC_MAX) if vec is None else vec,
        "partition" if batches * columns >= _C_THREAD_STAGE2 else "column",
    )
    size = rows * columns * itemsize
    # Anchor rows floor-divide whole MiB sizes; round back up to match them, so each anchor
    # also admits sizes up to 1 MiB below it.
    b200_size = -(-size // (1 << 20)) * (1 << 20)
    if (
        not explicit
        and cc == (10, 0)
        and dtype in (torch.float16, torch.bfloat16, torch.float32)
        and columns in (257, 4095)
        and b200_size in (16 << 20, 64 << 20, 256 << 20, 2 << 30)
        and acc_bits == 32
        and batches == 1
        and contiguous
        and alignment >= itemsize
    ):
        narrow = dtype in (torch.float16, torch.bfloat16)
        # Index and Welford keys come from internal kernel callers; the core overrides
        # route only sum, mean, prod, amax and amin.
        welford = nfields == 3 and welford_nouts(trait_key) == nouts
        index = (trait_key, nfields, nouts) == ("argmaxi32", 2, 1)
        one_field = (
            nfields == 1
            and nouts == 1
            and trait_key in ("sum", "prod", "mean", "amax", "amin", "vnorm2")
        )
        if columns == 4095 and b200_size == 16 << 20 and (one_field or welford):
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
                npar=npars[b200_size],
                vec=1,
                group=16 if narrow else 8,
            )
        if columns == 4095 and welford:
            return cfg._replace(
                rule="b200_ragged_welford_c4095",
                threads_per_block=64,
                npar=1928 if b200_size == 2 << 30 else 128,
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
                npar=npars[b200_size],
                vec=1,
            )
        if columns == 4095 and one_field:
            npar = (
                512
                if b200_size == 2 << 30 or b200_size == 256 << 20 and narrow
                else 256
            )
            return cfg._replace(
                rule="b200_ragged_c4095",
                threads_per_block=64,
                npar=npar,
                vec=1,
                partial_layout=(
                    "partition"
                    if narrow and b200_size == 256 << 20
                    else cfg.partial_layout
                ),
                combine_columns=(
                    8 if narrow and b200_size == 256 << 20 else cfg.combine_columns
                ),
                unroll=16 if narrow and b200_size == 256 << 20 else cfg.unroll,
            )
        if columns == 4095 and index and b200_size == 16 << 20 and not narrow:
            return cfg._replace(
                rule="b200_ragged_argmax_c4095_small",
                threads_per_block=32,
                npar=8,
                vec=1,
                unroll=64,
            )
        if columns == 4095 and index and b200_size >= 256 << 20:
            npar = (
                1024
                if narrow and b200_size == 2 << 30
                else 512
                if narrow or b200_size == 2 << 30
                else 128
            )
            return cfg._replace(
                rule="b200_ragged_argmax_c4095",
                threads_per_block=64,
                npar=npar,
                vec=1,
            )
    return cfg


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
        if worker_warps not in (1, 2, 4, 8, 16):
            raise ValueError("worker_warps must be one of 1, 2, 4, 8, 16")
        if tile_columns not in (8, 16, WARP):
            raise ValueError("tile_columns must be one of 8, 16, 32")
        self.trait = trait
        self.plan = plan
        self.R = R
        # C stays compile-time: on B200 a runtime row stride ran (16384, 1024) fp32 sums at
        # 0.66x. Batch counts are runtime, so only new (R, C) pairs compile.
        self.C = C
        self.nouts = nouts
        self.partition_begin = partition_begin
        self.partition_count = (
            plan.split[0] - partition_begin
            if partition_count is None and plan.shape == "split"
            else 1
            if partition_count is None
            else partition_count
        )
        if (
            self.partition_begin < 0
            or self.partition_count < 1
            or (
                plan.shape == "split"
                and self.partition_begin + self.partition_count > plan.split[0]
            )
        ):
            raise ValueError("ordered-column partition range is out of bounds")
        if plan.shape != "split" and (partition_begin, self.partition_count) != (0, 1):
            raise ValueError("only split plans have ordered-column partitions")
        # Each worker traverses an original warp tree, even with fewer column lanes.
        self.workers = min(worker_warps, plan.wpr)
        self.tile_columns = tile_columns
        # A ragged final split partition still needs address clamps.
        ragged_tail = (
            plan.shape == "split"
            and plan.split[1] != plan.split[2]
            and self.partition_begin + self.partition_count == plan.split[0]
        )
        self.full_tiles = full_tiles and _full_column_tiles(plan, R) and not ragged_tail

    @property
    def cache_sig(self) -> tuple[Any, ...]:
        key = (
            self.R,
            self.C,
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
        self, mIn: cute.Tensor, mOuts: list, in_base: Int64, num_o: Int32, stream
    ):
        self.kernel(mIn, mOuts, in_base, num_o).launch(
            grid=[cute.ceil_div(num_o, self.tile_columns), self.partition_count, 1],
            block=[self.tile_columns, self.workers, 1],
            stream=stream,
        )

    @cute.jit
    def _leaf(self, mIn, in_base, batch, col, red, active, bound):
        trait = self.trait
        valid = active & (red < bound)
        safe_red = red if const_expr(self.full_tiles) else red if valid else Int32(0)
        index = (
            in_base
            + Int64(batch) * Int64(self.R * self.C)
            + Int64(safe_red) * Int64(self.C)
            + Int64(col)
        )
        got = trait.leaf(mIn[index], safe_red)
        ident = trait.init()
        # Full tiles keep this select: without it norm2 leaves stop matching materialized
        # squares bitwise (test_traits.test_norm2_leaf_rounding).
        return tuple(got[f] if valid else ident[f] for f in range(trait.nfields))

    @cute.jit
    def _lane_root(self, mIn, in_base, batch, col, base, active, bound):
        vals = [
            self._leaf(
                mIn,
                in_base,
                batch,
                col,
                base + Int32(i),
                active,
                bound,
            )
            for i in range(self.plan.vec)
        ]
        return tile._reduce_vec(vals, self.plan.vec, self.trait.combine)

    @cute.kernel
    def kernel(self, mIn: cute.Tensor, mOuts: list, in_base: Int64, num_o: Int32):
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
        batch, col = safe // Int32(self.C), safe % Int32(self.C)
        nf = const_expr(trait.nfields)
        smem = cutlass.utils.SmemAllocator()
        roots = [
            smem.allocate_tensor(
                trait.fdtypes[f],
                cute.make_layout(const_expr(plan.wpr * columns)),
                byte_alignment=8,
            )
            for f in range(nf)
        ]
        loads, warp_chunk = plan.batches[0][2:]
        for warp in cutlass.range(worker, plan.wpr, self.workers):
            warp_base = split_base + warp * Int32(warp_chunk)
            bound = Int32(self.R)
            if const_expr(plan.shape == "split" and plan.split[1] != plan.split[2]):
                nbatch, bte, last, full_chunk, last_chunk = plan.split
                final_batch = partition == Int32(nbatch - 1)
                remaining = Int32(last) if final_batch else Int32(bte)
                chunk = Int32(last_chunk) if final_batch else Int32(full_chunk)
                offset = warp * chunk
                offset = remaining if offset > remaining else offset  # noqa: FURB136
                warp_base = split_base + offset
                end = offset + chunk
                end = remaining if end > remaining else end  # noqa: FURB136
                bound = split_base + end
            load_tree = []
            for load in cutlass.range_constexpr(loads):
                if const_expr(nf == 1):
                    lane_roots = cute.make_rmem_tensor(
                        cute.make_layout(WARP), trait.fdtypes[0]
                    )
                    for original_lane in cutlass.range_constexpr(WARP):
                        lane_roots[original_lane] = self._lane_root(
                            mIn,
                            in_base,
                            batch,
                            col,
                            warp_base
                            + Int32(load * WARP * plan.vec)
                            + Int32(original_lane * plan.vec),
                            active,
                            bound,
                        )[0]
                    load_root = tile._inner_tree_reduce(
                        [(lane_roots[i],) for i in range(WARP)], op
                    )
                else:
                    lane_roots = [
                        cute.make_rmem_tensor(cute.make_layout(WARP), trait.fdtypes[f])
                        for f in range(nf)
                    ]
                    for original_lane in cutlass.range(
                        WARP, unroll=_ORDERED_LANE_UNROLL
                    ):
                        root = self._lane_root(
                            mIn,
                            in_base,
                            batch,
                            col,
                            warp_base
                            + Int32(load * WARP * plan.vec)
                            + Int32(original_lane) * Int32(plan.vec),
                            active,
                            bound,
                        )
                        for f in cutlass.range_constexpr(nf):
                            lane_roots[f][original_lane] = root[f]
                    load_root = tile._inner_tree_reduce(
                        [
                            tuple(lane_roots[f][i] for f in range(nf))
                            for i in range(WARP)
                        ],
                        op,
                    )
                tile._streaming_push(load_tree, load_root, load, plan.depth, op)
            slot = warp * Int32(columns) + lane
            for f in cutlass.range_constexpr(nf):
                roots[f][slot] = load_tree[0][f]
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
            padding = ident
            # Constant identities fold Welford's 0 / 0 count ratio at trace time.
            if const_expr(nf > 1):
                dynamic_false = lane < Int32(0)
                padding = tuple(
                    group_vals[0][f] if dynamic_false else ident[f] for f in range(nf)
                )
            if const_expr(plan.shape != "split" or groups > 1):
                group_vals.extend([padding] * (WARP - groups))
            root = tile._inner_tree_reduce(group_vals, op)
            if const_expr(plan.shape == "split"):
                final = root
            else:
                # The indexed tree seeds its fold with the identity, so -0 sums become +0.
                final = op(ident, root)

        if const_expr(plan.shape == "split"):
            if (Int32(ty) == Int32(0)) & active:
                index = output * Int32(plan.split[0]) + partition
                for f in cutlass.range_constexpr(trait.nfields):
                    mOuts[f][index] = final[f]
        else:
            projected = trait.project(final, trait.acc(self.R))
            if (Int32(ty) == Int32(0)) & active:
                if const_expr(self.nouts == 1):
                    mOuts[0][output] = mOuts[0].element_type(projected)
                else:
                    for k in cutlass.range_constexpr(self.nouts):
                        mOuts[k][output] = mOuts[k].element_type(projected[k])


def _split_p(R: int) -> int:
    """Split the reduced axis into about _Q_TARGET rows per chunk."""
    return max(1, min(_P_MAX, -(-R // _Q_TARGET)))


def reduce_ordered_col(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtypes: Sequence[torch.dtype],
    nouts: int,
    *,
    allow_split: bool | None = None,
    full_tiles: bool | None = None,
) -> tuple[torch.Tensor, ...] | None:
    """Reduce the middle axis of a contiguous (R, C) or (B, R, C) physical view.

    allow_split=None lets the column policy pick the mapping and whether to run split
    plans; True runs split plans with default mappings and False declines them. Either
    bool bypasses the policy. An explicit full_tiles overrides the policy's, but with
    allow_split=None, split plans the policy does not select still decline.
    """
    if x.dim() == 2 and x.is_contiguous():
        B, (R, C) = 1, x.shape
        out_shape = (C,)
    elif x.dim() == 3 and x.is_contiguous():
        B, R, C = x.shape
        out_shape = (B, C)
    else:
        return None
    # The last block's Int32 output index must not wrap past the outputs.
    last = B * C + WARP
    if x.stride(-1) != 1 or C < WARP or last >= _INT32_LIMIT:
        return None
    plan = trait_itree_plan(
        trait, R, B * C, x.element_size(), stage=False, device=x.device
    )
    # Linear vector folds change the DAG, and this kernel replays tree folds only.
    if plan is None or plan.shape not in ("looped", "split") or plan.vec_linear:
        return None
    # Leaves fold plan.vec wide; a plan whose tiles load narrower would move bits.
    if plan.tms[0].vec != plan.vec:
        return None
    # Single-field traits replay all 32 lanes' leaves statically. Multi-field traits roll
    # lanes but replay a static lane tree per load, which compiles about as slowly (1.3 s
    # against 1.5 s at fp32 R = 8192). Cap both so long looped rows keep the indexed
    # tree's compile time. Plans past one planner batch always exceed the cap.
    if sum(b[2] for b in plan.batches) * WARP * plan.vec > tile.MAX_UNROLL:
        return None
    split = plan.shape == "split"
    # Split is verified bitwise for floating inputs only. Partials are indexed in Int32
    # and partitions ride grid y (at most 65535).
    floating = (torch.float32, torch.bfloat16, torch.float16, torch.float64)
    if split and (
        allow_split is False
        or x.dtype not in floating
        or B * C * plan.split[0] >= _INT32_LIMIT
        or plan.split[0] > 65535
    ):
        return None
    cfg = None
    if allow_split is None:
        cfg = select_ordered_col_config(
            _hw.caps(x.device).cc,
            x.dtype,
            trait_key,
            C,
            B,
            plan.split[0] if split else 1,
            field_bits=tuple(dt.width for dt in trait.fdtypes),
            out_dtypes=tuple(out_dtypes[:nouts]),
            rows=R,
            full_tiles=full_tiles is not False and _full_column_tiles(plan, R),
        )
        if cfg is None and split:
            return None
    workers, columns = _ORDERED_WORKER_WARPS, WARP
    if cfg is not None:
        workers, columns = cfg.worker_warps, cfg.tile_columns
        if full_tiles is None:
            full_tiles = cfg.full_tiles
    if split and full_tiles and plan.split[0] > 1 and plan.split[1] != plan.split[2]:
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
                full_tiles=False if full_tiles is None else full_tiles,
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
    fake_in = _L.fake_compact(torch2cute[kernel_x.dtype], (cute.sym_int64(),))
    fake_outs = [
        _L.fake_compact(torch2cute[out.dtype], (_L.sym(),)) for out in kernel_outs
    ]
    args = (Int64(x.storage_offset()), Int32(B * C), _stream())
    for op in ops:
        key = (
            "ordered_col1" if split else "ordered_col",
            trait_key,
            x.dtype,
            tuple(out.dtype for out in kernel_outs),
            str(x.device),
        ) + op.cache_sig
        build = lambda: _compile(op, fake_in, fake_outs, *args)  # noqa: E731
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
    cfg = select_col_config(
        _hw.caps(x.device).cc,
        x.dtype,
        R,
        C,
        B,
        trait.nfields,
        trait_key,
        itemsize=x.element_size(),
        acc_bits=trait.acc.width,
        nouts=nouts,
        alignment=align,
        contiguous=x.is_contiguous(),
        threads_per_block=threads_per_block,
        npar=npar,
        vec=vec,
    )
    if cfg.kernel_order not in ("linear", "inner_tree"):
        raise ValueError(f"unknown kernel order {cfg.kernel_order!r}")
    if cfg.kernel_order == "inner_tree":
        ordered = reduce_ordered_col(trait, trait_key, x, out_dtypes, nouts)
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
            _stream(),
        )

    # _VEC_MAX decouples vec from compile-time alignment, so key both.
    key = (
        "coltile",
        trait_key,
        x.dtype,
        tuple(out_dtypes[:nouts]),
        align,
        str(x.device),
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
        _stream(),
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
            str(x.device),
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
            _stream(),
        )
        ins = [_L.read_only(pp) for pp in parts]
        cached_plan(_CACHE, key2, build2, op=f"aten::{trait_key}")(
            ins, kernel_outs, Int32(total), nrows, Int32(npar), _stream()
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
            _stream(),
        )

    pdt = tuple(pp.dtype for pp in parts)
    key2 = (
        "coltile2",
        trait_key,
        x.dtype,
        tuple(out_dtypes[:nouts]),
        pdt,
        str(x.device),
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
        _stream(),
    )
    return tuple(outs)
