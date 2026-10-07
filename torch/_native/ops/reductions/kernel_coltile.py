# Column-reduction driver for tile.TileReduce, with launch policy and plan cache.
# Threads own vectorized kept-axis outputs without lane merging. Splitting the reduced
# axis provides parallelism; unsplit (65536, 256) took 7830us versus ATen's 15.8us.

import functools
import math
from collections.abc import Sequence
from typing import Any, NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64

import torch

from ...cutedsl import hw_caps as _hw, launch as _L
from ...cutedsl.dtypes import cute2torch, torch2cute
from ...cutedsl.plan_cache import cached_plan
from . import _storage, tile
from .kernel_general import _flat, _launch, ReduceBlock
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


class OrderedColConfig(NamedTuple):
    min_blocks: int
    worker_warps: int = _ORDERED_WORKER_WARPS
    tile_columns: int = WARP
    full_tiles: bool = False
    column_pack: int = 1


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
    if field_bits != (32,) * fields or out_dtypes != (output,) * nouts:
        return None
    if full_tiles and rows is not None:
        if rows == 65536 and columns == 1024 and fields != 2:
            cfg = cfg._replace(tile_columns=16, full_tiles=True)
        elif 32768 <= rows <= 131072 and 33 <= columns <= 257:
            minimum = 0 if fields == 3 else 16
            if dtype == torch.bfloat16 and partials >= 8:
                minimum //= 2
            width = WARP if partials >= 16 and columns == 257 else 16
            cfg = OrderedColConfig(minimum, tile_columns=width, full_tiles=True)
        elif rows == 16384 and columns == 4096 and fields == 3:
            cfg = cfg._replace(min_blocks=0, full_tiles=True)
        if dtype == torch.bfloat16 and columns == 4096:
            if rows == 32768 and trait_key == "amax":
                cfg = cfg._replace(tile_columns=16)
            elif rows == 262144 and trait_key in ("amax", "mean", "vnorm2"):
                pack = 2 if trait_key == "amax" else 4
                cfg = cfg._replace(tile_columns=WARP // pack, column_pack=pack)
    # Count useful column lanes, excluding padding in the last output tile.
    if partials * columns < WARP * cfg.min_blocks:
        return None
    return cfg


def _full_column_tiles(plan: Any, rows: int) -> bool:
    if not plan.batches:
        return False
    if plan.shape == "multirow":
        return plan.batches[0][2] * plan.vec <= rows
    limit = plan.split[1] if plan.shape == "split" else rows
    return all(
        off + (plan.wpr - 1) * chunk + loads * WARP * plan.vec <= limit
        for off, _, loads, chunk in plan.batches
    )


class ColConfig(NamedTuple):
    order: str
    rule: str
    threads_per_block: int
    npar: int
    vec: int
    partial_layout: str
    combine_columns: int = 0
    unroll: int = 4
    kernel_order: str = "linear"
    group: int = 1


@functools.lru_cache(maxsize=4096)
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
    order: str = "unordered",
    threads_per_block: int | None = None,
    npar: int | None = None,
    vec: int | None = None,
) -> ColConfig | None:
    """Select an implementation for unordered semantics, including ordered plans."""
    if order not in ("unordered", "inner_tree"):
        raise ValueError(f"unknown reduction order: {order!r}")
    if order == "inner_tree":
        return None
    explicit = any(v is not None for v in (threads_per_block, npar, vec))
    threads = _WIDE_ACC_THREADS_PER_BLOCK if nfields >= 3 else _THREADS_PER_BLOCK
    cfg = ColConfig(
        order,
        "explicit" if explicit else "default",
        threads if threads_per_block is None else threads_per_block,
        _split_p(rows) if npar is None else npar,
        min(tile.vec_size(columns, itemsize), _VEC_MAX) if vec is None else vec,
        "partition" if batches * columns >= _C_THREAD_STAGE2 else "column",
    )
    size = rows * columns * itemsize
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
    """Combine partition-major states while coalescing adjacent column loads."""

    def __init__(self, trait, columns, partitions, rows, tile_columns, nouts):
        if tile_columns not in (8, 16, 32):
            raise ValueError(f"combine columns must be 8, 16 or 32, got {tile_columns}")
        self.trait = trait
        self.columns = columns
        self.partitions = partitions
        self.rows = rows
        self.tile_columns = tile_columns
        self.threads = 128
        self.groups = self.threads // tile_columns
        self.nouts = nouts
        self.output_widths = tuple(
            getattr(trait, "output_widths", None) or (1,) * nouts
        )

    @property
    def cache_sig(self):
        return (
            self.columns,
            self.partitions,
            self.rows,
            self.tile_columns,
            self.nouts,
            self.output_widths,
        )

    @cute.jit
    def __call__(self, ins: list, outs: list, stream):
        self.kernel(ins, outs).launch(
            grid=[-(-self.columns // self.tile_columns), 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, ins: list, outs: list):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        lane = Int32(tx) % Int32(self.tile_columns)
        group = Int32(tx) // Int32(self.tile_columns)
        col = Int32(bx) * Int32(self.tile_columns) + lane
        safe_col = col if col < Int32(self.columns) else Int32(0)
        trait = self.trait
        combine, fdtypes = trait.combine, trait.fdtypes
        nf = const_expr(trait.nfields)
        identity = trait.init()
        acc = identity
        for p in cutlass.range(group, self.partitions, self.groups):
            offset = Int64(p) * Int64(self.columns) + Int64(safe_col)
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
        result = trait.project(acc, trait.acc(self.rows))
        if (group == Int32(0)) & (col < Int32(self.columns)):
            if const_expr(self.nouts == 1):
                tile._store_reduction_value(
                    outs[0], col, result, const_expr(self.output_widths[0])
                )
            else:
                for f in cutlass.range_constexpr(self.nouts):
                    tile._store_reduction_value(
                        outs[f], col, result[f], const_expr(self.output_widths[f])
                    )


class _OrderedColReduce:
    """Preserve the row inner-tree DAG while coalescing adjacent column loads."""

    def __init__(
        self,
        trait: Any,
        plan: Any,
        R: int,
        C: int,
        B: int,
        nouts: int,
        *,
        worker_warps: int = _ORDERED_WORKER_WARPS,
        tile_columns: int = WARP,
        full_tiles: bool = False,
        column_pack: int = 1,
        partition_begin: int = 0,
        partition_count: int | None = None,
    ) -> None:
        if plan.shape not in ("multirow", "looped", "split"):
            raise ValueError(f"ordered columns do not support {plan.shape=}")
        if plan.shape == "split" and plan.vec_linear:
            raise ValueError("ordered split columns require tree vector folds")
        if worker_warps not in (1, 2, 4, 8, 16):
            raise ValueError("worker_warps must be one of 1, 2, 4, 8, 16")
        if tile_columns not in (8, 16, WARP):
            raise ValueError("tile_columns must be one of 8, 16, 32")
        if column_pack not in (1, 2, 4):
            raise ValueError("column_pack must be one of 1, 2, 4")
        if column_pack > 1 and (
            trait.nfields != 1
            or trait.acc.width != 32
            or nouts != 1
            or C % column_pack
            or getattr(trait, "complex_input", False)
            or any(w != 1 for w in (getattr(trait, "output_widths", None) or (1,)))
        ):
            raise ValueError(
                "column packing requires complete single-field real outputs"
            )
        self.trait = trait
        self.plan = plan
        self.R = R
        self.C = C
        self.B = B
        self.nouts = nouts
        self.column_pack = column_pack
        self.output_groups = C // column_pack
        self.state_fields = trait.nfields * column_pack
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
        self.complex_input = getattr(trait, "complex_input", False)
        self.output_widths = (
            (1,) * nouts
            if plan.shape == "split"
            else tuple(getattr(trait, "output_widths", None) or (1,) * nouts)
        )
        # Each worker traverses an original warp tree, even with fewer column lanes.
        self.workers = 1 if plan.shape == "multirow" else min(worker_warps, plan.wpr)
        self.outputs_per_block = 128 if plan.shape == "multirow" else tile_columns
        self.full_tiles = full_tiles and _full_column_tiles(plan, R)

    @property
    def cache_sig(self) -> tuple[Any, ...]:
        key = (
            self.R,
            self.C,
            self.B,
            self.trait.nfields,
            self.nouts,
            self.output_widths,
            self.workers,
            self.outputs_per_block,
            self.full_tiles,
            self.column_pack,
            self.partition_begin,
            self.partition_count,
        )
        return key + (self.plan.sig,)

    @cute.jit
    def __call__(self, mIn: cute.Tensor, mOuts: list, in_base: Int64, stream):
        self.kernel(mIn, mOuts, in_base).launch(
            grid=[
                math.ceil(self.B * self.output_groups / self.outputs_per_block),
                self.partition_count,
                1,
            ],
            block=(
                [self.outputs_per_block, 1, 1]
                if const_expr(self.plan.shape == "multirow")
                else [self.outputs_per_block, self.workers, 1]
            ),
            stream=stream,
        )

    @cute.jit
    def _identity(self):
        if const_expr(self.column_pack > 1):
            return tuple(self.trait.init()[0] for _ in range(self.column_pack))
        return self.trait.init()

    @cute.jit
    def _combine(self, a, b):
        if const_expr(self.column_pack > 1):
            # Adjacent columns keep independent copies of the logical tree.
            return tuple(
                self.trait.combine((a[i],), (b[i],))[0] for i in range(self.column_pack)
            )
        return self.trait.combine(a, b)

    @cute.jit
    def _leaf(self, mIn, in_base, batch, col, red, active, bound):
        trait = self.trait
        valid = active & (red < bound)
        safe_red = red if const_expr(self.full_tiles) else red if valid else Int32(0)
        if const_expr(self.column_pack > 1):
            # Storage and columns are pack-aligned; keep row strides in pack units.
            index = (
                in_base // self.column_pack
                + Int64(batch) * Int64(self.R * (self.C // self.column_pack))
                + Int64(safe_red) * Int64(self.C // self.column_pack)
                + Int64(col) // self.column_pack
            )
            packed = cute.flat_divide(mIn, (self.column_pack,))
            frag = cute.make_rmem_tensor(
                cute.make_layout(self.column_pack), mIn.element_type
            )
            cute.autovec_copy(packed[None, index], frag)
            got = tuple(
                trait.leaf(frag[i], safe_red)[0] for i in range(self.column_pack)
            )
        else:
            index = (
                in_base
                + Int64(batch) * Int64(self.R * self.C)
                + Int64(safe_red) * Int64(self.C)
                + Int64(col)
            )
            got = trait.leaf(
                tile._load_reduction_value(
                    trait, mIn, index, const_expr(self.complex_input)
                ),
                safe_red,
            )
        ident = self._identity()
        return tuple(got[f] if valid else ident[f] for f in range(self.state_fields))

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
        return tile._reduce_vec(vals, self.plan.vec, self._combine)

    @cute.kernel
    def kernel(self, mIn: cute.Tensor, mOuts: list, in_base: Int64):
        tx, ty, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        trait, plan = self.trait, self.plan
        op, ident = self._combine, self._identity()
        split_base = Int32(0)
        partition = Int32(by) + Int32(const_expr(self.partition_begin))
        if const_expr(plan.shape == "split"):
            split_base = partition * Int32(plan.split[1])

        if const_expr(plan.shape == "multirow"):
            output = Int32(bx) * Int32(self.outputs_per_block) + Int32(tx)
            active = output < Int32(self.B * self.output_groups)
            safe = output if active else Int32(0)
            batch = safe // Int32(self.output_groups)
            col = (safe % Int32(self.output_groups)) * Int32(self.column_pack)
            tree = []
            for load in cutlass.range_constexpr(plan.batches[0][2]):
                tile._streaming_push(
                    tree,
                    self._lane_root(
                        mIn,
                        in_base,
                        batch,
                        col,
                        Int32(load * plan.vec),
                        active,
                        Int32(self.R),
                    ),
                    load,
                    plan.depth,
                    op,
                )
            final = tree[0]
        else:
            lane, worker = Int32(tx), Int32(ty)
            columns = const_expr(self.outputs_per_block)
            output = Int32(bx) * Int32(columns) + lane
            active = output < Int32(self.B * self.output_groups)
            safe = output if active else Int32(0)
            batch = safe // Int32(self.output_groups)
            col = (safe % Int32(self.output_groups)) * Int32(self.column_pack)
            nf = const_expr(self.state_fields)
            smem = cutlass.utils.SmemAllocator()
            roots = [
                smem.allocate_tensor(
                    trait.fdtypes[f % trait.nfields],
                    cute.make_layout(const_expr(plan.wpr * columns)),
                    byte_alignment=8,
                )
                for f in range(nf)
            ]
            final = ident
            for b in cutlass.range_constexpr(len(plan.batches)):
                batch_off, _remaining, loads, warp_chunk = plan.batches[b]
                for warp in cutlass.range(worker, plan.wpr, self.workers):
                    warp_base = split_base + Int32(batch_off) + warp * Int32(warp_chunk)
                    bound = Int32(self.R)
                    if const_expr(
                        plan.shape == "split" and plan.split[1] != plan.split[2]
                    ):
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
                            lane_tree = []
                            for original_lane in cutlass.range_constexpr(WARP):
                                tile._streaming_push(
                                    lane_tree,
                                    self._lane_root(
                                        mIn,
                                        in_base,
                                        batch,
                                        col,
                                        warp_base
                                        + Int32(load * WARP * plan.vec)
                                        + Int32(original_lane * plan.vec),
                                        active,
                                        bound,
                                    ),
                                    original_lane,
                                    5,
                                    op,
                                )
                            load_root = lane_tree[0]
                        tile._streaming_push(load_tree, load_root, load, plan.depth, op)
                    slot = warp * Int32(columns) + lane
                    for f in cutlass.range_constexpr(nf):
                        roots[f][slot] = load_tree[0][f]
                cute.arch.barrier()
                groups = const_expr(plan.wpr // plan.kchunk)
                if worker == Int32(0):
                    group_vals = [
                        tile._inner_tree_reduce(
                            [
                                tuple(
                                    roots[f][
                                        (g * plan.kchunk + i) * Int32(columns) + lane
                                    ]
                                    for f in range(nf)
                                )
                                for i in range(plan.kchunk)
                            ],
                            op,
                        )
                        for g in range(groups)
                    ]
                    padding = ident
                    if const_expr(nf > 1):
                        dynamic_false = lane < Int32(0)
                        padding = tuple(
                            group_vals[0][f] if dynamic_false else ident[f]
                            for f in range(nf)
                        )
                    if const_expr(plan.shape != "split" or groups > 1):
                        group_vals.extend([padding] * (WARP - groups))
                    root = tile._inner_tree_reduce(group_vals, op)
                    if const_expr(plan.shape == "split"):
                        final = root
                    else:
                        final = op(final, root)
                cute.arch.barrier()

        if const_expr(plan.shape == "split"):
            if (Int32(ty) == Int32(0)) & active:
                if const_expr(self.column_pack > 1):
                    for i in cutlass.range_constexpr(self.column_pack):
                        index = (output * Int32(self.column_pack) + Int32(i)) * Int32(
                            plan.split[0]
                        ) + partition
                        mOuts[0][index] = final[i]
                else:
                    index = output * Int32(plan.split[0]) + partition
                    for f in cutlass.range_constexpr(trait.nfields):
                        mOuts[f][index] = final[f]
        else:
            if const_expr(self.column_pack > 1):
                projected = tuple(
                    trait.project((final[i],), trait.acc(self.R))
                    for i in range(self.column_pack)
                )
            else:
                projected = trait.project(final, trait.acc(self.R))
            if (Int32(ty) == Int32(0)) & active:
                if const_expr(self.column_pack > 1):
                    for i in cutlass.range_constexpr(self.column_pack):
                        tile._store_reduction_value(
                            mOuts[0],
                            output * Int32(self.column_pack) + Int32(i),
                            projected[i],
                            1,
                        )
                elif const_expr(self.nouts == 1):
                    tile._store_reduction_value(
                        mOuts[0],
                        output,
                        projected,
                        const_expr(self.output_widths[0]),
                    )
                else:
                    for k in cutlass.range_constexpr(self.nouts):
                        tile._store_reduction_value(
                            mOuts[k],
                            output,
                            projected[k],
                            const_expr(self.output_widths[k]),
                        )


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
    order: str = "inner_tree",
    allow_split: bool | None = None,
    worker_warps: int | None = None,
    tile_columns: int | None = None,
    full_tiles: bool | None = None,
    column_pack: int | None = None,
) -> tuple[torch.Tensor, ...] | None:
    """Reduce the middle axis of a contiguous (B, R, C) physical view."""
    if order != "inner_tree":
        raise ValueError(f"ordered column reductions require inner_tree, got {order!r}")
    if x.dim() == 2:
        B, (R, C) = 1, x.shape
        out_shape = (C,)
    elif x.dim() == 3 and x.is_contiguous():
        B, R, C = x.shape
        out_shape = (B, C)
    else:
        return None
    if x.stride(-1) != 1 or C < WARP:
        return None
    if column_pack not in (None, 1, 2, 4):
        raise ValueError("column_pack must be one of 1, 2, 4")
    explicit_pack = column_pack is not None
    from . import kernel_rowtile as rt

    plan = rt.trait_itree_plan(
        trait, R, B * C, x.element_size(), stage=False, device=x.device
    )
    if plan is None:
        return None
    split = plan.shape == "split"
    if split and (
        allow_split is False
        or not x.is_contiguous()
        or x.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        or plan.vec_linear
        or B * C * plan.split[0] >= 2**31
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
    if column_pack is None:
        column_pack = 1 if cfg is None else cfg.column_pack
    if column_pack > 1 and (
        not x.is_contiguous()
        or x.dtype not in (torch.float32, torch.bfloat16)
        or x.storage_offset() % column_pack
        or _L.supported_alignment(_flat(x), column_pack * x.element_size())
        < column_pack * x.element_size()
    ):
        if explicit_pack:
            raise ValueError(
                "column packing requires aligned contiguous FP32/BF16 storage"
            )
        column_pack = 1
        if cfg is not None:
            cfg = cfg._replace(tile_columns=WARP)
    if cfg is not None:
        if worker_warps is None:
            worker_warps = cfg.worker_warps
        if tile_columns is None:
            tile_columns = cfg.tile_columns
        if full_tiles is None:
            full_tiles = cfg.full_tiles
    if worker_warps is None:
        worker_warps = _ORDERED_WORKER_WARPS
    op_kwargs = {
        "worker_warps": worker_warps,
        "tile_columns": WARP if tile_columns is None else tile_columns,
        "column_pack": column_pack,
    }
    if split and full_tiles and plan.split[0] > 1 and plan.split[1] != plan.split[2]:
        ops = (
            _OrderedColReduce(
                trait,
                plan,
                R,
                C,
                B,
                trait.nfields,
                full_tiles=True,
                partition_count=plan.split[0] - 1,
                **op_kwargs,
            ),
            _OrderedColReduce(
                trait,
                plan,
                R,
                C,
                B,
                trait.nfields,
                partition_begin=plan.split[0] - 1,
                partition_count=1,
                **op_kwargs,
            ),
        )
    else:
        ops = (
            _OrderedColReduce(
                trait,
                plan,
                R,
                C,
                B,
                trait.nfields if split else nouts,
                full_tiles=False if full_tiles is None else full_tiles,
                **op_kwargs,
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
        else [_storage.flat_view(out) for out in outs]
    )
    fake_in = _L.fake_compact(
        torch2cute[kernel_x.dtype],
        (_L.sym_int64(),),
        align=column_pack * x.element_size() if column_pack > 1 else None,
    )
    fake_outs = [
        _L.fake_compact(torch2cute[out.dtype], (_L.sym(),)) for out in kernel_outs
    ]
    for op in ops:
        key = (
            (
                "ordered_col1" if split else "ordered_col",
                trait_key,
                x.dtype,
                tuple(out_dtypes[:nouts]),
                str(x.device),
            )
            + op.cache_sig
            + (("order", order),)
        )
        build = lambda: _compile(  # noqa: E731
            op, fake_in, fake_outs, Int64(x.storage_offset()), _stream()
        )
        cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")(
            _L.read_only(kernel_x),
            kernel_outs,
            Int64(x.storage_offset()),
            _stream(),
        )
    if split:
        combine = rt.itree_combine_plan(
            plan,
            kernel_outs[0].element_size(),
            x.device,
            nfields=trait.nfields,
            nrows=B * C,
        )
        s2 = ReduceBlock(
            trait,
            count=plan.split[0],
            num_o=B * C,
            red_pairs=[],
            kept_pairs=[],
            project_n=R,
            nouts=nouts,
            order="inner_tree",
            itree=combine,
        )
        key2 = (
            "ordered_col2",
            trait_key,
            tuple(out_dtypes),
            tuple(part.dtype for part in kernel_outs),
        ) + s2.cache_sig
        _launch(s2, key2, kernel_outs, [_storage.flat_view(out) for out in outs])
    return tuple(outs)


def reduce_col_tile(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtype: torch.dtype,
    threads_per_block: int | None = None,
    npar: int | None = None,
    vec: int | None = None,
    *,
    order: str = "unordered",
) -> torch.Tensor:
    """Reduce dim 0 of contiguous 2D x to (C,), splitting it npar ways."""
    return _reduce_col_tile(
        trait,
        trait_key,
        x,
        [out_dtype],
        1,
        threads_per_block,
        npar,
        vec,
        order=order,
    )[0]


def reduce_col_tile_2out(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtypes: Sequence[torch.dtype],
    threads_per_block: int | None = None,
    npar: int | None = None,
    vec: int | None = None,
    *,
    order: str = "unordered",
) -> tuple[torch.Tensor, ...]:
    return _reduce_col_tile(
        trait,
        trait_key,
        x,
        out_dtypes,
        2,
        threads_per_block,
        npar,
        vec,
        order=order,
    )


def reduce_batched_col_tile(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtypes: Sequence[torch.dtype],
    nouts: int,
    *,
    order: str = "unordered",
) -> tuple[torch.Tensor, ...]:
    return _reduce_col_tile(
        trait, trait_key, x, out_dtypes, nouts, None, None, None, order=order
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
    *,
    order: str = "unordered",
) -> tuple[torch.Tensor, ...]:
    if order != "unordered":
        raise ValueError(f"column tiling requires unordered reduction, got {order!r}")
    if x.dim() not in (2, 3) or not x.is_cuda or x.stride(-1) != 1:
        raise AssertionError(
            f"want 2D/3D contiguous-last-dim CUDA, got {tuple(x.shape)}"
        )
    if x.dim() == 2:
        B, (R, C) = 1, x.shape
        rows = x
        out_shape = (C,)
    else:
        if not x.is_contiguous():
            raise AssertionError(f"want contiguous batched input, got {x.stride()}")
        B, R, C = x.shape
        rows = x.reshape(B * R, C)
        out_shape = (B, C)
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
        order=order,
        threads_per_block=threads_per_block,
        npar=npar,
        vec=vec,
    )
    if cfg is None:
        raise AssertionError("unordered column reduction needs a column configuration")
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
    kernel_x = _storage.row_view(rows)
    kernel_outs = [_storage.flat_view(out) for out in outs]
    total = B * C
    nchunks = Int32(total // vec)
    batch_chunks = Int32(C // vec) if x.dim() == 3 else None
    q, nrows = Int32(-(-R // npar)), Int32(R)

    single = npar == 1
    pc = cfg.partial_layout == "partition"
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
    kernel_dsts = [_storage.flat_view(dst) for dst in dsts]

    def _fake():
        # Dynamic descriptors require vec divisibility; None avoids a 1.27x unused argument.
        return (
            [
                _L.fake_compact(
                    torch2cute[kernel_x.dtype],
                    (_L.sym(), _L.sym(vec * (2 if x.is_complex() else 1))),
                    stride_order=(1, 0),
                    align=align,
                )
            ],
            [_L.fake_compact(torch2cute[d.dtype], (_L.sym(),)) for d in kernel_dsts],
            nchunks,
            batch_chunks,
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
        (
            "coltile",
            trait_key,
            x.dtype,
            tuple(out_dtypes[:nouts]),
            align,
            str(x.device),
        )
        + op.cache_sig
        + (("col_config", cfg),)
    )
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

    if cfg.combine_columns:
        op2 = PartialColumns(trait, total, npar, R, cfg.combine_columns, nouts)
        pdt = tuple(pp.dtype for pp in parts)
        key2 = (
            (
                "coltile2t",
                trait_key,
                tuple(out_dtypes[:nouts]),
                pdt,
                str(x.device),
            )
            + op2.cache_sig
            + (("col_config", cfg),)
        )
        build2 = lambda: _compile(  # noqa: E731
            op2,
            [_L.fake_compact(torch2cute[pp.dtype], (_L.sym_int64(),)) for pp in parts],
            [
                _L.fake_compact(torch2cute[out.dtype], (_L.sym(),))
                for out in kernel_outs
            ],
            _stream(),
        )
        cached_plan(_CACHE, key2, build2, op=f"aten::{trait_key}")(
            [_L.read_only(pp) for pp in parts], kernel_outs, _stream()
        )
        return tuple(outs)

    # Fold npar partials and project with true R; use threads per column only when C fills the GPU.
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
            (
                "coltile2b",
                trait_key,
                tuple(out_dtypes[:nouts]),
                pdt,
            )
            + s2.cache_sig
            + (("col_config", cfg),)
        )
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
        (
            "coltile2",
            trait_key,
            x.dtype,
            tuple(out_dtypes[:nouts]),
            pdt,
            str(x.device),
        )
        + op2.cache_sig
        + (("col_config", cfg),)
    )
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
