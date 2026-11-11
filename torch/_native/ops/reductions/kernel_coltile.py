# Column-reduction driver for tile.TileReduce, with launch policy and plan cache, plus the
# PartialColumns stage-two kernel and the ordered (inner-tree) column kernel.
# TileReduce threads own vectorized kept-axis outputs without lane merging. Splitting the
# reduced axis provides parallelism; unsplit (65536, 256) took 7830us versus ATen's 15.8us.
# Ordered column lanes each serve one output, and a block's warps split its rows as the row
# inner tree's warps do, loading rows strided by C; warp 0 merges their roots.
# Inputs must be on the current device.

from collections.abc import Sequence
from typing import Any, NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T as _T

import torch

from ...cutedsl import launch as _L
from ...cutedsl.dtypes import cute2torch, torch2cute
from ...cutedsl.plan_cache import cached_plan
from ...utils.device import check_current
from . import tile
from .kernel_general import (
    _combine_itree_partials,
    _flat,
    _INT32_LIMIT,
    _launch,
    ReduceBlock,
)
from .kernel_rowtile import trait_itree_plan
from .traits import WARP


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
# Multi-field lane replays partially unroll their 32-lane loop by this factor. On B200,
# fp32 argmax and bf16 var_mean columns (16, R, 1024) at R = 2048-8192 compiled in
# 0.4-1.2s instead of the static replay's 1.2-12.2s, running at 0.65-1.00x of its speed.
# Single-field replays stay static: before per-load stride copies (see _opaque_stride),
# rolled fp32 sums ran at as little as 0.46x the speed of static ones.
_ORDERED_LANE_UNROLL = 4


@dsl_user_op
def _opaque_stride(stride, load, *, loc=None, ip=None):
    """Widen a copy of the 32-bit stride that NVVM cannot relate to other loads' copies.

    NVVM's codegen time grows superlinearly with the runtime-stride addresses in one
    block; a copy per load keeps each load's addresses apart. Widening after the copy
    keeps each address one 32 x 32 -> 64-bit multiply-add."""
    copy = llvm.inline_asm(
        _T.i32(),
        [Int32(stride).ir_value(loc=loc, ip=ip)],
        f"mov.b32 $0, $1; // load {load}",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int64(Int32(copy))


class ColConfig(NamedTuple):
    rule: str
    threads_per_block: int
    npar: int
    vec: int
    partial_layout: str
    combine_columns: int = 0
    unroll: int = tile._ROLL_UNROLL


def select_col_config(
    rows: int,
    columns: int,
    batches: int,
    nfields: int,
    *,
    itemsize: int,
    threads_per_block: int | None = None,
    npar: int | None = None,
    vec: int | None = None,
) -> ColConfig:
    """Select column geometry and partial layout; explicit launch arguments override the geometry.

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
    ) -> None:
        if plan.shape not in ("looped", "split"):
            raise ValueError(f"ordered columns do not support {plan.shape=}")
        if plan.vec_linear:
            raise ValueError("ordered columns replay tree vector folds only")
        if len(plan.batches) != 1:
            raise ValueError("ordered columns fold a single planner batch")
        self.trait = trait
        self.plan = plan
        self.R = R
        # C and the batch count are runtime, so each R compiles once per stride width:
        # row views take a 64-bit stride once R * C needs one, so a new C can recompile.
        # Each leaf costs an address multiply-add that a compile-time C folds into the
        # load, but the clamp keeps addresses independent: against per-C kernels, B200
        # sum, argmax and var_mean (fp32/bf16) ran at 1.04x geomean (0.79-1.70x) for
        # looped R 128-8192 and 1.18x (0.69-1.68x) for split R 32768-131072, and a new C
        # skips a 0.8 s (median) compile.
        self.wide = R * C >= _INT32_LIMIT
        self.nouts = nouts
        # Split plans launch one block row per partition.
        self.partitions = plan.split[0] if plan.shape == "split" else 1
        # Exact looped plans tile the row without padded warps or leaves past R. Skipping
        # their leaf masks, and the shared-memory merge for one-warp rows, ran B200 argmax
        # at up to 1.4x and var_mean at up to 2.4x on exact views; sums were within noise.
        self.exact = plan.shape == "looped" and plan.tms[0].exact
        # Exact single-field replays of several loads address each load through its own
        # stride copy (see _opaque_stride) and ask for two blocks per SM, without which
        # ptxas front-loads more loads at lower occupancy.
        self.per_load = trait.nfields == 1 and plan.batches[0][2] > 1 and self.exact

    @property
    def cache_sig(self) -> tuple[Any, ...]:
        return (self.R, self.wide, self.trait.nfields, self.nouts, self.plan.sig)

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
            grid=[cute.ceil_div(num_o, WARP), self.partitions, 1],
            block=[WARP, self.plan.wpr, 1],
            stream=stream,
            min_blocks_per_mp=2 if self.per_load else 0,
        )

    @cute.jit
    def _leaf(self, rows, warp_base, k, active, bound, last):
        trait = self.trait
        red = warp_base + k
        if const_expr(self.exact):
            # Every leaf is in bounds, and inactive lanes never store their outputs.
            return trait.leaf(rows[cutlass.min(Int32(k), last)], red)
        valid = active & (red < bound)
        # Leaves past the bound read its last in-bounds row; the select below discards
        # them. The clamp also keeps each address independent of the last one.
        got = trait.leaf(rows[cutlass.min(Int32(k), last)], red if valid else Int32(0))
        ident = trait.init()
        return tuple(got[f] if valid else ident[f] for f in range(trait.nfields))

    @cute.jit
    def _lane_root(self, rows, replay, warp_base, lane, load, active, bound, last):
        vals = [
            self._leaf(rows, warp_base, replay((i, lane, load)), active, bound, last)
            for i in range(self.plan.vec)
        ]
        return tile._reduce_vec(vals, self.plan.vec, self.trait.combine)

    @cute.kernel
    def kernel(
        self, mIn: cute.Tensor, mOuts: list, in_base: Int64, num_o: Int32, C: Int32
    ):
        tx, ty, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        trait, plan = self.trait, self.plan
        op, ident = trait.combine, trait.init()
        split_base = Int32(0)
        partition = Int32(by)
        if const_expr(plan.shape == "split"):
            split_base = partition * Int32(plan.split[1])

        # Each warp of threads replays one original warp's tree.
        lane, warp = Int32(tx), Int32(ty)
        output = Int32(bx) * Int32(WARP) + lane
        active = output < num_o
        safe = output if active else Int32(0)
        # The (B, R, C) input as ((C, B), R): flat outputs index its first mode, so
        # tiles cross batches unless C is a multiple of their width. Batches always
        # take a 64-bit stride.
        row_stride = Int64(C) if const_expr(self.wide) else C
        view = cute.make_tensor(
            mIn.iterator + in_base,
            cute.make_layout(
                ((C, num_o // C), self.R),
                stride=((1, Int64(self.R) * Int64(C)), row_stride),
            ),
        )
        nf = const_expr(trait.nfields)
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
        rows = cute.domain_offset(first, view[safe, None])
        last = cutlass.max(bound - Int32(1) - first, Int32(0))
        # (vector element, original lane, load) -> row of the view, element fastest.
        replay = cute.make_layout((plan.vec, WARP, loads))
        load_tree = []
        for load in cutlass.range_constexpr(loads):
            if const_expr(self.per_load):
                by_load = cute.make_layout(self.R, stride=_opaque_stride(C, load))
                rows = cute.make_tensor(rows.iterator, by_load)
            lane_roots = [
                cute.make_rmem_tensor(cute.make_layout(WARP), dtype)
                for dtype in trait.fdtypes
            ]
            # Single-field traits replay lanes statically; multi-field ones roll them.
            if const_expr(nf == 1):
                for orig_lane in cutlass.range_constexpr(WARP):
                    root = self._lane_root(
                        rows, replay, warp_base, orig_lane, load, active, bound, last
                    )
                    lane_roots[0][orig_lane] = root[0]
            else:
                for orig_lane in cutlass.range(WARP, unroll=_ORDERED_LANE_UNROLL):
                    root = self._lane_root(
                        rows, replay, warp_base, orig_lane, load, active, bound, last
                    )
                    for f in cutlass.range_constexpr(nf):
                        lane_roots[f][orig_lane] = root[f]
            lane_vals = [tuple(t[i] for t in lane_roots) for i in range(WARP)]
            load_root = tile._inner_tree_reduce(lane_vals, op)
            tile._streaming_push(load_tree, load_root, load, plan.depth, op)
        if const_expr(plan.wpr == 1):
            # One warp replays the whole row, so its root needs no shared-memory merge.
            # Only looped plans run one warp; like the indexed tree, seed the identity
            # so -0 sums become +0.
            final = op(ident, load_tree[0])
        else:
            groups = const_expr(plan.wpr // plan.kchunk)
            # (column, (warp in chunk, group)): each original warp's roots are contiguous.
            by_warp = cute.make_layout(
                (WARP, (plan.kchunk, groups)), stride=(1, (WARP, WARP * plan.kchunk))
            )
            smem = cutlass.utils.SmemAllocator()
            roots = [
                smem.allocate_tensor(dtype, by_warp, byte_alignment=8)
                for dtype in trait.fdtypes
            ]
            for f in cutlass.range_constexpr(nf):
                roots[f][lane, warp] = load_tree[0][f]
            cute.arch.barrier()
            final = ident
            if warp == Int32(0):
                group_vals = [
                    tile._inner_tree_reduce(
                        [
                            tuple(t[lane, (i, g)] for t in roots)
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


def reduce_ordered_col(
    trait: Any,
    trait_key: str,
    x: torch.Tensor,
    out_dtypes: Sequence[torch.dtype],
    nouts: int,
    *,
    allow_split: bool = False,
) -> tuple[torch.Tensor, ...] | None:
    """Reduce axis -2 of a contiguous (R, C) or (B, R, C) view, or return None."""
    if not x.is_cuda:
        raise AssertionError(f"want a CUDA input, got {x.device}")
    device = x.get_device()
    check_current(device)
    if x.dim() not in (2, 3) or not x.is_contiguous():
        return None
    B, R, C = (1, *x.shape) if x.dim() == 2 else x.shape
    out_shape = (*x.shape[:-2], C)
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
    # Single-field traits replay all 32 lanes' leaves statically, compiling in 0.35 s at
    # R = 2048 (fp32/bf16, C = 2048). With per-load stride copies, exact replays of 512
    # leaves per warp (R = 8192) compile in 1.5-2.4 s instead of 9.2-10.2 s (B200
    # fp32/bf16/fp16 sums, fp32/bf16 amax). Multi-field traits roll lanes but replay a
    # static lane tree per load, taking 1.3 s at R = 8192 and 2.5-3.1 s at
    # R = 8193-16384 (fp32). The indexed tree takes 0.3-0.75 s, so cap replays at
    # tile.MAX_UNROLL leaves (looped rows up to 8192).
    # Plans past one planner batch always exceed the cap.
    if sum(b[2] for b in plan.batches) * WARP * plan.vec > tile.MAX_UNROLL:
        return None
    split = plan.shape == "split"
    # Split is verified bitwise for floating inputs only. Partials are indexed in Int32
    # and partitions ride grid y.
    floating = (torch.float32, torch.bfloat16, torch.float16, torch.float64)
    if split and (
        not allow_split
        or x.dtype not in floating
        or B * C * plan.split[0] >= _INT32_LIMIT
        or plan.split[0] > _MAX_NPAR
    ):
        return None
    op = _OrderedColReduce(trait, plan, R, C, trait.nfields if split else nouts)
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
    args = (Int64(x.storage_offset()), Int32(B * C), Int32(C), _stream(device))
    key = (
        "ordered_col1" if split else "ordered_col",
        trait_key,
        x.dtype,
        tuple(out.dtype for out in kernel_outs),
        device,
    ) + op.cache_sig

    def build():
        fake_in = _L.fake_compact(torch2cute[kernel_x.dtype], (cute.sym_int64(),))
        fake_outs = [
            _L.fake_compact(torch2cute[out.dtype], (_L.sym(),)) for out in kernel_outs
        ]
        return _compile(op, fake_in, fake_outs, *args)

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
    cfg = select_col_config(
        R,
        C,
        B,
        trait.nfields,
        itemsize=x.element_size(),
        threads_per_block=threads_per_block,
        npar=npar,
        vec=vec,
    )
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
