# TWO-STAGE cross-CTA row reduction for few-row / huge-N (and the M=1 reduce-all case),
# mirroring ATen's Reduce.cuh. Stage 1 reshapes (M, N) -> (M*C, N/C) and reduces each sub-row
# to a RAW accumulator with no cross-block sync; stage 2 combines each row's C partials and
# projects once with the true N, which keeps mean/var correct. 1.13-1.66x of ATen.
#
# C comes from N alone, not from M or the SM count, because it is baked into the plan. INDEX
# traits are DECLINED: the reshape makes a sub-row's chunk index row % C, so they go to
# kernel_general's ragged split, whose gidx_from="chunk" carries the absolute index.

import math
from typing import NamedTuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Float64, Int32, Int64

import torch

from ...cutedsl.dtypes import torch2cute
from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan
from . import (  # safe: kernel_general imports us only lazily
    kernel_general as _RB,
    kernel_rowtile as _rt,
)


_PART_TORCH = {Float32: torch.float32, Float64: torch.float64, Int32: torch.int32}


_C_MAX = 1 << 22  # cap stage-2 partial count (its combine is a dynamic loop -> cheap)
# Target stage-1 sub-row length. The reshape puts a whole sub-row in ONE block's tile, so it
# trades directly against occupancy: at the smem ceiling ~1 block/SM and ~3.5 TB/s, at 8-20k
# several blocks/SM and ~7.3. Below ~4k the stage-2 combine starts to cost. 8192 is the flat
# top of that curve for throughput and for compile time.
_SUBROW_TARGET = 8192
# Stage-2 threads-per-block: the combine is bandwidth-light (folds C partials), so a
# mid block is plenty; parameterized so the autotuner / a retune can override.
_DEFAULT_BLOCK = 256


class _XctaConfig(NamedTuple):
    # xcta's own knob set; a knob the caller leaves None takes the default here. subrow_target is the
    # stage-1 sub-row length TARGET, which _split_C snaps to a legal C by nearest-divisor search. No
    # nfields term: wide-accumulator traits prefer shorter sub-rows, but the shift is shape-dependent
    # and no closed form captures it without regressing nfields=1.
    block: int = _DEFAULT_BLOCK
    subrow_target: int = _SUBROW_TARGET


def _split_C(N, vec, smem_budget_elems, subrow_target=None):
    # C = chunks per output row. The reshape needs C | N exactly, with the sub-row length a
    # multiple of vec that fits its tile. We do NOT maximize it -- that pins ~1 block/SM and halves
    # bandwidth -- but aim for subrow_target and take the nearest divisor, searching outward. The
    # search is bounded by the smem budget rather than by N, which matters when the divisors near
    # the target lie tens of thousands apart.
    target = _SUBROW_TARGET if subrow_target is None else subrow_target
    step = max(vec, 1)
    # Floor on sub-row length: below it stage 1 barely reduces (a prime N degenerates to N
    # partials and a no-op stage 1). Those stay on the grid-striding fallback instead.
    lo = max(step, 256)
    hi = min(smem_budget_elems, N)
    hi -= hi % step
    if hi < lo:
        return None
    tgt = min(max(target - (target % step), lo), hi)
    # Expand symmetrically from tgt; first divisor of N (that keeps C <= _C_MAX) wins.
    for d in range(0, hi - lo + step, step):
        for s in (tgt + d, tgt - d):
            # C == 1 is rejected: a split into one chunk IS the one-shot, which every caller has already
            # declined. It arises for a PRIME N, whose only in-window divisor is N itself -- (8, 65537)
            # bf16 then folded the whole sub-row at vec=1, 74us against ATen's 6.0.
            if lo <= s <= hi and N % s == 0 and 1 < N // s <= _C_MAX:
                return N // s
    return None


class FusedTwoStage:
    # BOTH stage launches in ONE @cute.jit region: one compile artifact and one host-side call, so
    # the Python dispatch and arg marshalling are paid once. The two launches still serialize on
    # the stream. This replicates the two objects' own launch bodies back to back.
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
        # --- stage 1 launch (mirrors RowTile.__call__) ---
        # The tile row kernel with final=False: it writes the RAW per-field accumulator of
        # each sub-row. Its fold is ROLLED, so the sub-row length arrives as runtime args
        # (nchunks/nwaves) and distinct N in a vec class share ONE compiled kernel.
        s1.kernel(mX, parts, s1_nchunks, s1_nwaves, project_n).launch(
            grid=[cute.ceil_div(mX.shape[0], const_expr(s1.rows_per_block)), 1, 1],
            block=[const_expr(s1.nt), 1, 1],
            stream=stream,
        )
        # --- stage 2: one block per output row, with the grid read live so the fused kernel serves any
        # M. A single kept dim means the decode ignores the extent, and the geometry arrives as RUNTIME
        # args, so one kernel serves every N in the vec class with its own C. ---
        s2 = self.s2
        s2.kernel(
            parts,
            mOuts,
            [cute.FastDivmodDivisorV2(e) for e in rexts],
            rstrides,
            [cute.FastDivmodDivisorV2(e) for e in kexts],
            kstrides,
            count,
            cutlass.Int64(0),
            cutlass.Int64(count),
            project_n,
        ).launch(grid=[mOuts[0].shape[0], 1, 1], block=[s2.block, 1, 1], stream=stream)


# Two-level cache: compiled kernels keyed on the sub-row's vec class and stage-1 config, and
# per-(N, knobs) derivations including the pre-boxed args (~6us of boxing) and None declines.
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
    # Two-output form: stage 2 projects BOTH fields of the same combined accumulator, so the split
    # costs nothing over nouts==1. None if the split is declined.
    return _reduce_row_xcta(
        trait, trait_key, x, list(out_dtypes), 2, block, flatten, subrow_target
    )


def _reduce_row_xcta(
    trait, trait_key, x, out_dtypes, nouts, block, flatten, subrow_target
):
    # FUSED two-stage row reduction; flatten collapses an M==1 result to 0-d, which a 1-D
    # reduce-ALL wants and a (1, N) reduce-DIM does not. The single @cute.jit region is what turns
    # this regime from ~0.6x of ATen with two launches into ~1.15-1.44x, and it captures cleanly.
    if not (x.is_cuda and x.is_contiguous()):
        raise AssertionError(
            f"need a contiguous CUDA input, got {x.device} {x.stride()}"
        )
    if x.dim() == 1:
        x = x.view(1, -1)
    M, N = x.shape
    # Fill the knobs from the config when the caller passed None; explicit values pass through.
    cfg = _XctaConfig()
    block = cfg.block if block is None else block
    subrow_target = cfg.subrow_target if subrow_target is None else subrow_target

    # DYNAMIC M: the plan depends on N, dtype and trait but NOT M, so one kernel serves any batch
    # size and M only sizes the grid and scratch. Stage 1 casts its tile coordinate to Int64, so a
    # past-2**31 offset does not overflow. Two cache levels, as above, with subrow_target in the
    # geometry key because it changes C.
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

    # One fused launch. Scratch is sized per call since M varies, and every operand is dynamic-M
    # so the cached kernel serves any M.
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
    # resize_ (not view) so the 0-d result is not a VIEW of `out`: aten reductions
    # never alias, and view-ness is observable (see kernel_general._as_shape).
    if flatten and M == 1:
        return tuple(o.resize_(()) for o in outs)
    return tuple(outs)


def _build_geom(trait, trait_key, x, out_dtypes, nouts, M, N, block, subrow_target):
    # Derive the C split for this N, then compile or reuse the fused kernel for its bucket. None
    # declines; the caller memoizes that, so the fallback is remembered too.
    device = x.device
    elsize = x.element_size()
    vec = math.gcd(N, 128 // (elsize * 8))
    # C is fixed in the plan (it sets stage-1's sub-row length N//C and stage-2's partial
    # count), so the same C must serve every M -> derive it once here, from N alone.
    C = _split_C(N, vec, _RB._SMEM_BUDGET // elsize, subrow_target)
    if C is None:
        # Prime / poorly-factored N: no clean reshape split. The caller memoizes the
        # None -> the K0 general kernel serves it (any N, no reshape, O(1) compile).
        return None
    s = N // C
    # INDEX traits are declined: stage 1 would have to rebase its within-sub-row column to the
    # global one, which the reshape makes awkward. The ragged split serves them at 1.29-3.17x of
    # ATen, since its gidx_from="chunk" already carries the absolute index.
    if getattr(trait, "has_index", False):
        return None
    svec = math.gcd(s, 128 // (elsize * 8))  # the sub-row's OWN vec class

    # ONE stage-1 shape: the row kernel's fold is ROLLED, so the sub-row length is a runtime arg
    # and one compiled kernel serves the whole vec class. N is absent from the key for that reason.
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
    # Stage 1's rolled-loop counts over its sub-row: nchunks vec-groups, tpr per wave.
    s1_counts = (Int32(s // svec), Int32(-(-(s // svec) // tpr)))

    def _make_s1():
        return _rt.RowTile(trait, torch2cute[x.dtype], s, tpr, nt, nouts, False, unroll)

    def _fake_in():
        # 2D row-major, both extents dynamic: mode 1 divisible by the sub-row vec width, so one
        # compiled kernel serves every sub-row length in that vec class.
        return _L.fake_compact(
            torch2cute[x.dtype], (_L.sym(), _L.sym(svec)), order=(1, 0), align=align
        )

    def _fake_1d(dtype):
        return _L.fake_compact(torch2cute[dtype], (_L.sym(),))

    def _build():
        s1 = _make_s1()
        # Stage 2's geometry values are runtime launch args; the object contributes only the
        # STRUCTURAL cache_sig fields, so the compiled kernel is geometry-agnostic.
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
        # Descriptors only -- no seed tensors to allocate, and every extent dynamic so the compiled
        # fn serves any M.
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
    # Pre-boxed stage-2 runtime args, memoized because boxing is us-scale. The kept pair carries a
    # seed M, but a single-pair decode ignores the extent, so M stays fully dynamic.
    return (
        _RB._exts([(C, 1)]),
        _RB._strides([(C, 1)]),
        _RB._exts([(M, C)]),
        _RB._strides([(M, C)]),
        Int32(C),
        Int64(N),
    )
