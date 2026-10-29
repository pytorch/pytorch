# Column-reduction driver for tile.TileReduce, with launch policy and plan cache.
# Threads own vectorized kept-axis outputs without lane merging. Splitting the reduced
# axis provides parallelism; unsplit (65536, 256) took 7830us versus ATen's 15.8us.

from collections.abc import Sequence
from typing import Any, NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64

import torch

from ...cutedsl import launch as _L
from ...cutedsl.dtypes import cute2torch, torch2cute
from ...cutedsl.plan_cache import cached_plan
from . import tile
from .kernel_general import _INT32_LIMIT, _launch, ReduceBlock


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


class ColConfig(NamedTuple):
    rule: str
    threads_per_block: int
    npar: int
    vec: int
    partial_layout: str
    combine_columns: int = 0
    unroll: int = 4


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
    """Select stage-one column geometry; explicit launch arguments override it.

    Whether a view reaches the column kernel at all is `kernel_general._col_route_ok`'s call.
    """
    explicit = any(v is not None for v in (threads_per_block, npar, vec))
    threads = _WIDE_ACC_THREADS_PER_BLOCK if nfields >= 3 else _THREADS_PER_BLOCK
    # Partials are indexed in Int32 up to outputs * npar.
    outputs = batches * columns
    max_npar = _INT32_LIMIT // max(1, outputs)
    if npar is not None and npar > max_npar:
        raise ValueError(f"npar={npar} overflows Int32 partials for {outputs} outputs")
    cfg = ColConfig(
        "explicit" if explicit else "default",
        threads if threads_per_block is None else threads_per_block,
        min(_split_p(rows), max_npar) if npar is None else npar,
        min(tile.vec_size(columns, itemsize), _VEC_MAX) if vec is None else vec,
        "partition" if batches * columns >= _C_THREAD_STAGE2 else "column",
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

    @property
    def cache_sig(self):
        return (
            self.columns,
            self.partitions,
            self.rows,
            self.tile_columns,
            self.nouts,
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
                outs[0][col] = outs[0].element_type(result)
            else:
                for f in cutlass.range_constexpr(self.nouts):
                    outs[f][col] = outs[f].element_type(result[f])


def _split_p(R: int) -> int:
    """Split the reduced axis into about _Q_TARGET rows per chunk."""
    return max(1, min(_P_MAX, -(-R // _Q_TARGET)))


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
            [
                _L.fake_compact(torch2cute[pp.dtype], (cute.sym_int64(),))
                for pp in parts
            ],
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
