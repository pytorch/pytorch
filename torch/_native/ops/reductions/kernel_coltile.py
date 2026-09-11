# Column-reduction driver for tile.TileReduce, with launch policy and plan cache.
# Threads own vectorized kept-axis outputs without lane merging. Splitting the reduced
# axis provides parallelism; unsplit (65536, 256) took 7830us versus ATen's 15.8us.

from cutlass import Int32

import torch

from ...cutedsl import launch as _L
from ...cutedsl.dtypes import torch2cute
from ...cutedsl.plan_cache import cached_plan
from . import tile
from .kernel_general import _launch, _PART_TORCH, ReduceBlock


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
# Small blocks avoid idle threads on narrow columns; nt=256 took 17.3us versus 9.9us
# at 64. Use 32 threads for 1-2 fields and 64 for register-heavy Welford. Against the
# pre-shared kernel this is 0.92-1.01x, except (16384, 1024) sum/amax lose 7-9%,
# while argmax gains 6-8% and wide-short sum gains 15%.
_NT = 32
_NT_WIDE_ACC = 64  # 3-field traits (Welford): see above


def _split_p(R):
    """Split the reduced axis into about _Q_TARGET rows per chunk."""
    return max(1, min(_P_MAX, -(-R // _Q_TARGET)))


def reduce_col_tile(trait, trait_key, x, out_dtype, nt=None, npar=None, vec=None):
    """Reduce dim 0 of contiguous 2D x to (C,), splitting it npar ways."""
    if x.dim() != 2 or not x.is_cuda or x.stride(-1) != 1:
        raise AssertionError(f"want 2D contiguous-last-dim CUDA, got {tuple(x.shape)}")
    if nt is None:
        nt = _NT_WIDE_ACC if trait.nfields >= 3 else _NT
    R, C = x.shape
    vec = min(tile.vec_size(C, x.element_size()), _VEC_MAX) if vec is None else vec
    if C % vec:
        # An explicit nondivisor vec would leave trailing outputs uninitialized.
        raise AssertionError(f"vec must divide the column count: {C=} {vec=}")
    if npar is None:
        npar = _split_p(R)
    out = torch.empty(C, device=x.device, dtype=out_dtype)
    align = tile.align_bytes(C, x.element_size())
    nchunks, q, nrows = Int32(C // vec), Int32(-(-R // npar)), Int32(R)

    single = npar == 1
    pc = C >= _C_THREAD_STAGE2  # (P, C) for a thread-per-column stage 2, else (C, P)
    op = tile.TileReduce(
        trait,
        torch2cute[x.dtype],
        "col",
        C,
        nt=nt,
        final=single,
        vec=vec,
        pc=pc,
    )
    parts = (
        []
        if single
        else [
            torch.empty(C * npar, device=x.device, dtype=_PART_TORCH[trait.fdtypes[f]])
            for f in range(trait.nfields)
        ]
    )
    dsts = [out] if single else parts

    def _fake():
        # Dynamic descriptors require vec divisibility; None avoids a 1.27x unused argument.
        return (
            [
                _L.fake_compact(
                    torch2cute[x.dtype],
                    (_L.sym(), _L.sym(vec)),
                    stride_order=(1, 0),
                    align=align,
                )
            ],
            [_L.fake_compact(torch2cute[d.dtype], (_L.sym(),)) for d in dsts],
            nchunks,
            None,  # nwaves: the row axis's
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
    key = ("coltile", trait_key, x.dtype, out_dtype, align) + op.cache_sig
    build = lambda: _compile(op, *_fake())  # noqa: E731
    cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")(
        [_L.read_only(x)],
        list(dsts),
        nchunks,
        None,
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
        return out

    # Fold npar partials and project with true R; use threads per column only when C fills the GPU.
    if not pc:
        s2 = ReduceBlock(
            trait,
            count=npar,
            num_o=C,
            red_pairs=[(npar, 1)],
            kept_pairs=[(C, npar)],
            from_partials=True,
            project_n=R,
            nouts=1,
            final=True,
            block=128,
        )
        pdt = tuple(pp.dtype for pp in parts)
        key2 = ("coltile2b", trait_key, out_dtype, pdt) + s2.cache_sig
        _launch(s2, key2, parts, [out])
        return out

    # Shared combine mode uses nchunks as C and nrows as true R; q is unused.
    op2 = tile.TileReduce(
        trait, torch2cute[x.dtype], "col", C, nt=nt, vec=1, combine=True
    )

    def _fake2():
        # Row nwaves and split q are unused.
        return (
            [_L.fake_compact(torch2cute[pp.dtype], (_L.sym(),)) for pp in parts],
            [_L.fake_compact(torch2cute[out.dtype], (_L.sym(),))],
            Int32(C),
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
    key2 = ("coltile2", trait_key, x.dtype, out_dtype, pdt) + op2.cache_sig
    build2 = lambda: _compile(op2, *_fake2())  # noqa: E731
    cached_plan(_CACHE, key2, build2, op=f"aten::{trait_key}")(
        [_L.read_only(pp) for pp in parts],
        [out],
        Int32(C),
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
    return out
