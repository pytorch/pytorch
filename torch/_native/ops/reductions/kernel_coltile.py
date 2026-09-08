# COLUMN reduction (dim 0 of a contiguous 2D input): a DRIVER over tile.TileReduce, owning the
# measured launch policy and the plan cache. It differs from the row case in two ways -- one
# output per THREAD with no lane merge, and vectorization along the KEPT axis. The REDUCED axis
# must be split or the reduction carries no parallelism: unsplit, (65536, 256) took 7830us
# against ATen's 15.8.

from cutlass import Int32

import torch

from ...cutedsl.dtypes import torch2cute
from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan
from . import tile
from .kernel_general import _launch, _PART_TORCH, ReduceBlock


_compile = _L.compile_kernel
_stream = _L.stream
_CACHE = {}

# Rows per chunk of the reduced axis. MEASURED: the optimal split factor is a constant ~64
# ROWS per chunk across every shape, not a constant P -- capping P instead left the
# tall-narrow case at a quarter of the throughput.
_Q_TARGET = 64
_P_MAX = 4096
# Stage 2's work mapping. Block-per-column wins while C is small enough that C blocks is not
# itself the cost, thread-per-column once C/nt alone fills the device; the crossover measured
# between 4096 and 16384.
_C_THREAD_STAGE2 = 8192
# Columns per thread. Here `vec` sets the load width AND the live ACCUMULATOR count, a tension
# the row case does not have. Capped at 4: bf16 at 8 is 0.77-0.83x of 4.
_VEC_MAX = 4
# Threads per block, SMALL on purpose: a block covers nt column-chunks, so a wide block idles
# most of its threads whenever the column count is short, and the reduced-axis split already
# supplies blocks. Measured 17.3us at nt=256 against 9.9 at 64 on the tall-narrow case.
#
# 32 for 1- and 2-field traits and 64 for 3-field, from an interleaved A/B against the
# pre-shared kernel: a Welford accumulator is register-heavy enough to want a second warp per
# block to hide latency, while the lean traits want the narrower one. That pair holds the
# merged body at 0.92-1.01x of the pre-merge kernel except (16384, 1024) sum/amax, which lose
# 7-9%, against argmax gaining 6-8% and a wide-short sum 15%.
_NT = 32
_NT_WIDE_ACC = 64  # 3-field traits (Welford): see above


def _split_p(R):
    """Chunks of the reduced axis, from the measured ~_Q_TARGET-rows-per-chunk rule."""
    return max(1, min(_P_MAX, -(-R // _Q_TARGET)))


def reduce_col_tile(trait, trait_key, x, out_dtype, nt=None, npar=None, vec=None):
    """Reduce dim 0 of a contiguous 2D `x` -> (C,), splitting the reduced axis npar ways."""
    if x.dim() != 2 or not x.is_cuda or x.stride(-1) != 1:
        raise AssertionError(f"want 2D contiguous-last-dim CUDA, got {tuple(x.shape)}")
    if nt is None:
        nt = _NT_WIDE_ACC if trait.nfields >= 3 else _NT
    R, C = x.shape
    vec = min(tile.vec_size(C, x.element_size()), _VEC_MAX) if vec is None else vec
    if C % vec:
        # nchunks = C // vec, so a trailing partial group would never be stored and `out` would keep
        # whatever torch.empty gave it. The derived vec always divides C; an explicit one may not.
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
        # Compile-time descriptors; both extents dynamic, the inner one divisible by vec. The row
        # axis's arg is None, not a dummy -- an unused Int32 param costs 1.27x here.
        return (
            [
                _L.fake_compact(
                    torch2cute[x.dtype],
                    (_L.sym(), _L.sym(vec)),
                    order=(1, 0),
                    align=align,
                )
            ],
            [_L.fake_compact(torch2cute[d.dtype], (_L.sym(),)) for d in dsts],
            nchunks,
            None,  # nwaves: the row axis's
            nrows,
            q,
            Int32(npar),
            None,  # rvals, kvals, in_base, limit: the general axis's decode
            None,
            None,
            None,
            _stream(),
        )

    # align is baked in by _compile, and the _VEC_MAX cap means equal vec no longer implies
    # equal alignment the way the row path's uncapped vec does -- so key it.
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
        _stream(),
    )
    if single:
        return out

    # Stage 2: fold each column's npar partials and project once with the TRUE reduced
    # extent -- thread-per-column when C alone fills the device, else block-per-column.
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

    # Stage 2 is the SAME body in combine mode: nchunks carries the column count (one
    # thread each), nrows the true reduced extent for project, and q is unused.
    op2 = tile.TileReduce(
        trait, torch2cute[x.dtype], "col", C, nt=nt, vec=1, combine=True
    )

    def _fake2():
        # nchunks carries the column count (one thread each), project_n the true reduced extent
        # for the projection; the row axis's nwaves and the split's q are unused -> None.
        return (
            [_L.fake_compact(torch2cute[pp.dtype], (_L.sym(),)) for pp in parts],
            [_L.fake_compact(torch2cute[out.dtype], (_L.sym(),))],
            Int32(C),
            None,  # nwaves
            Int32(R),
            None,  # q
            Int32(npar),
            None,  # rvals, kvals, in_base, limit
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
        _stream(),
    )
    return out
