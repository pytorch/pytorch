# Two-stage cross-CTA row reduction for few rows with large N, mirroring ATen Reduce.cuh.
# Stage 1 reshapes (M, N) to (M*C, N/C) and emits raw accumulators; stage 2 combines C
# partials and projects with true N, preserving mean/variance. C depends only on N, so plans
# reuse across M. Reshaping obscures global indices, so index traits use the general kernel's
# ragged split. Measured 1.13-1.66x of ATen.

import math
from typing import NamedTuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Float64, Int32, Int64

import torch

from ...cutedsl import launch as _L
from ...cutedsl.dtypes import torch2cute
from ...cutedsl.plan_cache import cached_plan
from . import (  # safe: kernel_general imports us only lazily
    kernel_general as _RB,
    kernel_rowtile as _rt,
    tile,
)


_PART_TORCH = {Float32: torch.float32, Float64: torch.float64, Int32: torch.int32}


_C_MAX = 1 << 22  # cap stage-2 partial count (its combine is a dynamic loop -> cheap)
# Target one-block sub-row length. Smem-limited rows reach ~3.5 TB/s, 8-20k reach
# ~7.3 TB/s with several blocks/SM, and below ~4k stage 2 costs; 8192 balances throughput
# and compile time.
_SUBROW_TARGET = 8192
# Stage 2 only folds C partials, so a tunable mid-sized block suffices.
_DEFAULT_BLOCK = 256


class _XctaConfig(NamedTuple):
    # Defaults for omitted knobs. _split_C snaps subrow_target to a nearby legal divisor.
    # Wide accumulators can prefer shorter rows, but no nfields formula avoids regressions.
    block: int = _DEFAULT_BLOCK
    subrow_target: int = _SUBROW_TARGET


def _split_C(N, vec, max_subrow_elems, subrow_target=None):
    # Find C | N whose vector-aligned sub-row fits the tile and is nearest the target.
    # Maximizing sub-row size can halve bandwidth by leaving ~1 block/SM. Bound the search
    # by the sub-row cap because nearby divisors may be far apart.
    target = _SUBROW_TARGET if subrow_target is None else subrow_target
    step = max(vec, 1)
    # Below 256 elements, stage 1 barely reduces; use the grid-striding fallback.
    lo = max(step, 256)
    hi = min(max_subrow_elems, N)
    hi -= hi % step
    if hi < lo:
        return None
    tgt = min(max(target - (target % step), lo), hi)
    # Search outward from the target for the first divisor with bounded C.
    for d in range(0, hi - lo + step, step):
        for s in (tgt + d, tgt - d):
            # Reject the already-declined C == 1 one-shot. For prime (8, 65537) bf16,
            # scalar folding took 74us versus ATen's 6.0us.
            if lo <= s <= hi and N % s == 0 and 1 < N // s <= _C_MAX:
                return N // s
    return None


class FusedTwoStage:
    # Fuse both serialized launches into one compilation and host call, paying Python
    # dispatch and argument marshalling once.
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        parts: list,
        mOuts: list,
        rexts: list,
        rstrides: list,
        kexts: list,
        kstrides: list,
        count: cutlass.Int32,
        project_n: cutlass.Int64,
        s1_nchunks: cutlass.Int32,
        s1_nwaves: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        s1 = self.s1
        # Stage 1 emits raw accumulators. Runtime loop counts share a kernel across N;
        # wide sub-rows coalesce directly, so omit TMA and unused axis arguments.
        s1.kernel(
            [mX],
            parts,
            None,
            s1_nchunks,
            s1_nwaves,
            project_n,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], const_expr(s1.rows_per_block)), 1, 1],
            block=[const_expr(s1.nt), 1, 1],
            stream=stream,
        )
        # Stage 2 uses one block per output row. Runtime grid and geometry let one
        # kernel serve every M and every N in the vector class with its own C.
        s2 = self.s2
        # Follow shared-body argument order; None omits costly row/column arguments.
        s2.tile.kernel(
            parts,
            mOuts,
            None,
            count,
            None,
            project_n,
            None,
            None,
            [cute.FastDivmodDivisorV2(e) for e in rexts],
            rstrides,
            [cute.FastDivmodDivisorV2(e) for e in kexts],
            kstrides,
            cutlass.Int64(0),
            cutlass.Int64(count),
        ).launch(grid=[mOuts[0].shape[0], 1, 1], block=[s2.block, 1, 1], stream=stream)


# Cache kernels by vector class/config and geometry, boxed arguments (~6us), and declines by N.
_PLAN = {}
_GEOM = {}


def reduce_row_xcta(
    trait, trait_key, x, out_dtype, block=None, flatten=False, subrow_target=None
):
    res = _reduce_row_xcta(
        trait, trait_key, x, [out_dtype], 1, block, flatten, subrow_target
    )
    return None if res is None else res[0]


def reduce_row_xcta_2out(
    trait, trait_key, x, out_dtypes, block=None, flatten=False, subrow_target=None
):
    # Project both fields from one accumulator at no extra split cost; None means declined.
    return _reduce_row_xcta(
        trait, trait_key, x, list(out_dtypes), 2, block, flatten, subrow_target
    )


def _reduce_row_xcta(
    trait, trait_key, x, out_dtypes, nouts, block, flatten, subrow_target
):
    # flatten makes M == 1 scalar for reduce-all, unlike reduce-dim. Fusing both launches
    # improved ~0.6x of ATen to ~1.15-1.44x and remains graph-capturable.
    if not (x.is_cuda and x.is_contiguous()):
        raise AssertionError(
            f"need a contiguous CUDA input, got {x.device} {x.stride()}"
        )
    if x.dim() == 1:
        x = x.view(1, -1)
    M, N = x.shape
    cfg = _XctaConfig()
    block = cfg.block if block is None else block
    subrow_target = cfg.subrow_target if subrow_target is None else subrow_target

    # M only sizes grid and scratch, so one plan serves any batch size. Stage 1 uses Int64
    # tile coordinates beyond 2**31. subrow_target belongs in the geometry key because it changes C.
    out_dtypes = tuple(out_dtypes)
    gkey = (trait_key, x.dtype, out_dtypes, N, block, subrow_target, str(x.device))
    geom = _GEOM.get(gkey)
    if geom is None and gkey not in _GEOM:
        geom = _GEOM[gkey] = _build_geom(
            trait, trait_key, x, out_dtypes, nouts, M, N, block, subrow_target
        )
    if geom is None:  # memoized refusal (prime / poorly-factored N) -> K0
        return None
    C, s, fn, rexts, rstrides, kexts, kstrides, cnt, pn, s1nc, s1nw = geom

    # Size scratch per M; all operands keep M dynamic for plan reuse.
    sub = x.reshape(M * C, s)
    parts = [
        torch.empty(M * C, device=x.device, dtype=_PART_TORCH[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    outs = [torch.empty(M, device=x.device, dtype=d) for d in out_dtypes]
    fn(
        _L.read_only(sub),
        list(parts),
        list(outs),
        rexts,
        rstrides,
        kexts,
        kstrides,
        cnt,
        pn,
        s1nc,
        s1nw,
        _L.stream(),
    )
    # resize_ preserves ATen's non-aliasing scalar output; view-ness is observable.
    if flatten and M == 1:
        return tuple(o.resize_(()) for o in outs)
    return tuple(outs)


def _build_geom(trait, trait_key, x, out_dtypes, nouts, M, N, block, subrow_target):
    # Derive C, then compile or reuse its bucket; memoized None records a decline.
    device = x.device
    elsize = x.element_size()
    vec = math.gcd(N, 128 // (elsize * 8))
    # C fixes both stages' geometry, so derive it from N alone for reuse across M.
    C = _split_C(N, vec, _RB._MAX_ROW_BYTES // elsize, subrow_target)
    if C is None:
        # Prime or poorly factored N uses K0's reshape-free, extent-independent fallback.
        return None
    s = N // C
    # Decline index traits because reshaping obscures global columns; the ragged split
    # already carries them and measured 1.29-3.17x of ATen.
    if getattr(trait, "has_index", False):
        return None
    svec = math.gcd(s, 128 // (elsize * 8))  # the sub-row's OWN vec class

    # Runtime rolled-loop length lets one stage-1 kernel serve a vector class, so omit N.
    cfg = _rt.row_config(s, elsize * 8)
    tpr = max(_rt.WARP, cfg.tpr)
    nt = max(tpr, cfg.nt)
    nt -= nt % tpr  # rows_per_block must be whole
    unroll = 16 if svec == 1 else 4  # scalar sub-rows want more loads in flight
    pkey = (
        "xcta",
        trait_key,
        x.dtype,
        out_dtypes,
        svec,
        tpr,
        nt,
        unroll,
        block,
        str(device),
    )
    align = svec * elsize
    # Stage-1 rolled-loop counts: vector groups, then waves of tpr.
    s1_counts = (Int32(s // svec), Int32(-(-(s // svec) // tpr)))

    def _make_s1():
        return tile.TileReduce(
            trait,
            torch2cute[x.dtype],
            "row",
            s,
            tpr=tpr,
            nt=nt,
            nouts=nouts,
            final=False,
            unroll=unroll,
        )

    def _fake_in():
        # Dynamic 2D row-major descriptor, with inner extent divisible by vector width.
        return _L.fake_compact(
            torch2cute[x.dtype],
            (_L.sym(), _L.sym(svec)),
            stride_order=(1, 0),
            align=align,
        )

    def _fake_1d(dtype):
        return _L.fake_compact(torch2cute[dtype], (_L.sym(),))

    def _build():
        s1 = _make_s1()
        # Runtime stage-2 geometry keeps the compiled kernel geometry-independent.
        s2 = _RB.ReduceBlock(
            trait,
            count=C,
            num_o=M,
            red_pairs=[(C, 1)],
            kept_pairs=[(M, C)],
            from_partials=True,
            project_n=N,
            nouts=nouts,
            final=True,
            block=block,
        )
        fop = FusedTwoStage(s1, s2)
        # Dynamic descriptors avoid seed tensors and serve any M.
        return _L.compile_kernel(
            fop,
            _fake_in(),
            [_fake_1d(_PART_TORCH[trait.fdtypes[f]]) for f in range(trait.nfields)],
            [_fake_1d(d) for d in out_dtypes],
            *_s2_args(C, M, N),
            *s1_counts,
            _L.stream(),
        )

    fn = cached_plan(_PLAN, pkey, _build, op=f"aten::{trait_key}")
    return (C, s, fn, *_s2_args(C, M, N), *s1_counts)


def _s2_args(C, M, N):
    # Memoize boxed stage-2 arguments; single-pair decode ignores seed M, keeping it dynamic.
    return (
        _RB._exts([(C, 1)]),
        _RB._strides([(C, 1)]),
        _RB._exts([(M, C)]),
        _RB._strides([(M, C)]),
        Int32(C),
        Int64(N),
    )
