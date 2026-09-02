# COLUMN reduction (dim 0 of a contiguous 2D input): a DRIVER over tile.TileReduce at axis="col",
# owning the measured launch policy and the plan cache. Differs from the row case in two ways: one
# output per THREAD with no lane merge, and vectorization along the KEPT axis (tile.fold_cols_rolled).
# The REDUCED axis must be split P ways or the reduction carries no parallelism -- unsplit,
# (65536, 256) took 7830us against ATen's 15.8. The partial LAYOUT follows stage 2's mapping: (P, C)
# thread-per-column, (C, P) block-per-column.

from cutlass import Int32

import torch

from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan
from . import tile
from .kernel_general import _launch, _PART_TORCH, ReduceBlock


_compile = _L.compile_kernel
_stream = _L.stream
_CACHE = {}

# Rows per chunk of the reduced axis. MEASURED: the optimal split factor across (4096,4096),
# (16384,1024), (65536,256) and (1024,16384) is P = 64/256/1024/16 -- i.e. a constant ~64 ROWS
# per chunk in every case, not a constant P. Capping P at 64 instead (the first cut) left the
# tall-narrow (65536,256) at a quarter of the throughput the rows-per-chunk rule reaches.
_Q_TARGET = 64
_P_MAX = 4096
# Stage 2's work mapping, chosen like stage 1's and for the same reason. thread-per-column
# gives C threads (C/nt blocks); block-per-column gives C blocks. MEASURED (us, thread- vs
# block-per-column): C=256 82.4/15.7, C=1024 28.1/10.6, C=4096 15.2/11.7, C=16384 11.5/18.1,
# C=65536 10.5/16.6 -- so block wins while C is small enough that C blocks is not itself the
# cost, and thread wins once C/nt alone fills the device. Crossover bracketed 4096..16384.
_C_THREAD_STAGE2 = 8192
# Columns per thread. For a COLUMN reduction `vec` sets the load width AND the live ACCUMULATOR
# count per thread, a tension the row case does not have. Capped at 4 because the registers a wider
# width costs outweigh the load: bf16 at vec=8 is 0.77-0.83x of vec=4. A no-op for fp32 and fp64.
_VEC_MAX = 4
# Threads per block. SMALL on purpose: a block covers nt column-chunks, so a wide block idles
# most of its threads whenever the column count is short (C=256 at vec=4 is 64 chunks -- 64 of
# 256 threads busy), and the reduced-axis split already supplies blocks. The tall-narrow case
# is where that matters most: bf16 (65536,256) measured 17.3us at nt=256 vs 9.9 at 64.
#
# RE-MEASURED once the body became shared with the row axis. Timings on this box drift over a
# long run, so this is an INTERLEAVED A/B (both kernels in one process, alternating rounds,
# minimum of 4) of the pre-shared kernel at nt=64 against the shared one at 32 and at 64:
#
#   shape          op      nf   pre@64   new@32   new@64
#   (65536, 256)   sum      1    10.52    10.66    12.16
#   (65536, 256)   amax     1    13.12    13.23    14.05
#   (65536, 256)   argmax   2    31.08    28.71    29.49
#   (65536, 256)   var      3    34.79    41.33    34.85
#   (16384, 1024)  sum      1    11.03    12.04    12.56
#   (16384, 1024)  argmax   2    31.57    29.66    29.98
#   (16384, 1024)  var      3    33.29    33.84    33.27
#   (4096, 4096)   sum      1    12.41    12.52    14.38
#   (4096, 4096)   var      3    38.91    38.85    38.89
#   (256, 65536)   sum      1    11.25     9.58     9.09
#
# So 32 for 1- and 2-field traits and 64 for 3-field: a Welford accumulator is `vec` x 3
# values per thread, register-heavy enough that it wants a second warp per block to hide
# latency, while the lean traits want the narrower block. That pair holds the merged body at
# 0.92-1.01x of the pre-merge kernel everywhere except (16384, 1024) sum/amax, which lose
# 7-9%; argmax gains 6-8% and a wide-short (256, 65536) sum gains 15%.
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
        # nchunks = C // vec, so the trailing C % vec columns would never be stored and
        # `out` would keep whatever torch.empty gave them. The derived vec always divides
        # C; an explicit one is the caller's, so check it.
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
        _L.torch2cute[x.dtype],
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

    def _wrap():
        # nwaves belongs to the ROW axis: None, not a dummy -- an unused Int32 param costs
        # 1.27x here (see tile.TileReduce.kernel). project_n carries the reduced extent,
        # which is also what the per-block chunk bound is measured against.
        return (
            [_L.cute_tensor_dynMN(x, vec, align=align, read_only=True)],
            [_L.cute_tensor_dynM(d, ndim=1) for d in dsts],
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
    build = lambda: _compile(op, *_wrap())  # noqa: E731
    cached_plan(_CACHE, key, build, op=f"aten::{trait_key}")(*_wrap())
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
        trait, _L.torch2cute[x.dtype], "col", C, nt=nt, vec=1, combine=True
    )

    def _wrap2():
        # nchunks carries the column count (one thread each), project_n the true reduced extent
        # for the projection; the row axis's nwaves and the split's q are unused -> None.
        return (
            [_L.cute_tensor_dynM(pp, ndim=1) for pp in parts],
            [_L.cute_tensor_dynM(out, ndim=1)],
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
    build2 = lambda: _compile(op2, *_wrap2())  # noqa: E731
    cached_plan(_CACHE, key2, build2, op=f"aten::{trait_key}")(*_wrap2())
    return out
