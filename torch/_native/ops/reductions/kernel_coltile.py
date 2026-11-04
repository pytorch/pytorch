# Column-reduction driver for tile.TileReduce, with launch policy and plan cache, plus the
# PartialColumns stage-two kernel and the ordered (inner-tree) column kernel.
# TileReduce threads own vectorized kept-axis outputs without lane merging. Splitting the
# reduced axis provides parallelism; unsplit (65536, 256) took 7830us versus ATen's 15.8us.
# Ordered column lanes each serve one output, and a block's warps split its rows as the row
# inner tree's warps do, loading rows strided by C; warp 0 merges their roots.

from collections.abc import Sequence
from typing import Any, NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64

import torch

from ...cutedsl import launch as _L
from ...cutedsl.dtypes import cute2torch, torch2cute
from ...cutedsl.plan_cache import cached_plan
from ...utils.device import check_current
from . import tile
from .kernel_general import _flat, _INT32_LIMIT, _launch, ReduceBlock
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
# Stage one launches chunks along grid y.
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
# fp32 argmax (2, 32769, 33) and bf16 var_mean (2, 32768, 33) columns compiled in
# 0.8-1.2s instead of the static replay's 7.8-8.6s and ran at 0.92-1.06x of it.
# Single-field replays stay static: rolled, fp32 sums ran at as little as 0.46x.
_ORDERED_LANE_UNROLL = 4


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
        if plan.shape != "looped":
            raise ValueError(f"ordered columns do not support {plan.shape=}")
        if plan.vec_linear:
            raise ValueError("ordered columns replay tree vector folds only")
        if len(plan.batches) != 1:
            raise ValueError("ordered columns fold a single planner batch")
        self.trait = trait
        self.plan = plan
        self.R = R
        # C and the batch count are runtime, so each R compiles once. Each leaf costs an
        # address multiply-add that a compile-time C folds into the load, but the clamp
        # keeps addresses independent: against per-C kernels, B200 sum, argmax and
        # var_mean (fp32/bf16, R 128-8192) ran at 1.04x geomean (0.79-1.70x), and a
        # new C skips a 0.8 s (median) compile. Row views take a 64-bit stride once
        # R * C needs one.
        self.wide = R * C >= _INT32_LIMIT
        self.nouts = nouts
        # Exact plans tile the row without padded warps or leaves past R. Skipping their
        # leaf masks, and the shared-memory merge for one-warp rows, ran B200 argmax at up
        # to 1.4x and var_mean at up to 2.4x on exact views; sums were within noise.
        self.exact = plan.tms[0].exact

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
            grid=[cute.ceil_div(num_o, WARP), 1, 1],
            block=[WARP, self.plan.wpr, 1],
            stream=stream,
        )

    @cute.jit
    def _leaf(self, rows, warp_base, k, active, last):
        trait = self.trait
        red = warp_base + k
        if const_expr(self.exact):
            # Every leaf is in bounds, and inactive lanes never store their outputs.
            return trait.leaf(rows[cutlass.min(Int32(k), last)], red)
        valid = active & (red < Int32(self.R))
        # Leaves past R read the view's last in-bounds row, row R - 1; the select below
        # discards them. The clamp also keeps each address independent of the last one.
        got = trait.leaf(rows[cutlass.min(Int32(k), last)], red if valid else Int32(0))
        ident = trait.init()
        return tuple(got[f] if valid else ident[f] for f in range(trait.nfields))

    @cute.jit
    def _lane_root(self, rows, warp_base, k, active, last):
        vals = [
            self._leaf(rows, warp_base, k + i, active, last)
            for i in range(self.plan.vec)
        ]
        return tile._reduce_vec(vals, self.plan.vec, self.trait.combine)

    @cute.kernel
    def kernel(
        self, mIn: cute.Tensor, mOuts: list, in_base: Int64, num_o: Int32, C: Int32
    ):
        tx, ty, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        trait, plan = self.trait, self.plan
        op, ident = trait.combine, trait.init()

        # Each warp of threads replays one original warp's tree.
        lane, warp = Int32(tx), Int32(ty)
        output = Int32(bx) * Int32(WARP) + lane
        active = output < num_o
        safe = output if active else Int32(0)
        batch, col = safe // C, safe % C
        col_base = in_base + Int64(batch) * Int64(self.R) * Int64(C) + Int64(col)
        nf = const_expr(trait.nfields)
        loads, warp_chunk = plan.batches[0][2:]
        warp_base = warp * Int32(warp_chunk)
        # A padded warp can start at R; its leaves all mask, so its view starts at R-1.
        first = warp_base if warp_base < Int32(self.R) else Int32(self.R - 1)
        rows = cute.make_tensor(
            mIn.iterator + col_base + Int64(first) * Int64(C),
            cute.make_layout(
                const_expr(loads * WARP * plan.vec),
                stride=Int64(C) if const_expr(self.wide) else C,
            ),
        )
        last = Int32(self.R - 1) - first
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
                        last,
                    )
                    for f in cutlass.range_constexpr(nf):
                        lane_roots[f][original_lane] = root[f]
                load_root = tile._inner_tree_reduce(
                    [tuple(lane_roots[f][i] for f in range(nf)) for i in range(WARP)],
                    op,
                )
            tile._streaming_push(load_tree, load_root, load, plan.depth, op)
        # The indexed tree seeds its fold with the identity, so -0 sums become +0.
        if const_expr(plan.wpr == 1):
            # One warp replays the whole row, so its root needs no shared-memory merge.
            final = op(ident, load_tree[0])
        else:
            smem = cutlass.utils.SmemAllocator()
            roots = [
                smem.allocate_tensor(
                    trait.fdtypes[f],
                    cute.make_layout(const_expr(plan.wpr * WARP)),
                    byte_alignment=8,
                )
                for f in range(nf)
            ]
            slot = warp * Int32(WARP) + lane
            for f in cutlass.range_constexpr(nf):
                roots[f][slot] = load_tree[0][f]
            cute.arch.barrier()
            groups = const_expr(plan.wpr // plan.kchunk)
            final = ident
            if warp == Int32(0):
                group_vals = [
                    tile._inner_tree_reduce(
                        [
                            tuple(
                                roots[f][(g * plan.kchunk + i) * Int32(WARP) + lane]
                                for f in range(nf)
                            )
                            for i in range(plan.kchunk)
                        ],
                        op,
                    )
                    for g in range(groups)
                ]
                # Like TileReduce._fold_itree, only several groups merge across 32 lanes.
                root = group_vals[0]
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
                    root = tile._inner_tree_reduce(group_vals, op)
                final = op(ident, root)

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
) -> tuple[torch.Tensor, ...] | None:
    """Reduce axis -2 of a contiguous (R, C) or (B, R, C) view, or return None."""
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
    if plan is None or plan.shape != "looped" or plan.vec_linear:
        return None
    # Single-field traits replay all 32 lanes' leaves statically. Multi-field traits roll
    # lanes but replay a static lane tree per load, which compiles about as slowly (1.3 s
    # against 1.5 s at fp32 R = 8192). Cap both so long looped rows keep the indexed
    # tree's compile time. Plans past one planner batch always exceed the cap.
    if sum(b[2] for b in plan.batches) * WARP * plan.vec > tile.MAX_UNROLL:
        return None
    op = _OrderedColReduce(trait, plan, R, C, nouts)
    device = x.get_device()
    check_current(device)
    outs = [
        torch.empty(out_shape, device=x.device, dtype=dtype)
        for dtype in out_dtypes[:nouts]
    ]
    kernel_x = _flat(x)
    kernel_outs = [out.reshape(-1) for out in outs]

    def _fake():
        fake_in = _L.fake_compact(torch2cute[kernel_x.dtype], (cute.sym_int64(),))
        fake_outs = [
            _L.fake_compact(torch2cute[out.dtype], (_L.sym(),)) for out in kernel_outs
        ]
        return fake_in, fake_outs

    args = (Int64(x.storage_offset()), Int32(B * C), Int32(C), _stream(device))
    key = (
        "ordered_col",
        trait_key,
        x.dtype,
        tuple(out_dtypes[:nouts]),
        device,
    ) + op.cache_sig
    build = lambda: _compile(op, *_fake(), *args)  # noqa: E731
    cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")(
        _L.read_only(kernel_x), kernel_outs, *args
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
