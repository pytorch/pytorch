# TWO-STAGE cross-CTA row reduction for few-row / huge-N (and the M=1 reduce-all case), mirroring
# ATen's Reduce.cuh. Stage 1 reshapes (M, N) -> (M*C, N/C) and has kernel_rowtile reduce each sub-row
# to a RAW accumulator (final=False), with no cross-block sync, so it runs at full bandwidth; stage 2
# COMBINES each row's C partials one block per row and projects once with the true N, which is what
# keeps mean/var correct and gives the 2-output split for free. 5.4 TB/s on (1,16M) fp32 = 1.13x
# ATen, fp16 up to 1.66x.
#
# C comes from N alone (nearest divisor to a measured sub-row target that still fits the tile, see
# _split_C), deliberately not from M or the SM count, because C is baked into the plan. INDEX traits
# are DECLINED: the reshape makes a sub-row's chunk index row % C; they go to
# kernel_general._two_stage_row, whose gidx_from="chunk" carries the absolute index.

import math
from typing import NamedTuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Float64, Int32, Int64

import torch

from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan
from . import (  # safe: kernel_general imports us only lazily
    kernel_general as _RB,
    kernel_rowtile as _rt,
)


_PART_TORCH = {Float32: torch.float32, Float64: torch.float64, Int32: torch.int32}


_C_MAX = 1 << 22  # cap stage-2 partial count (its combine is a dynamic loop -> cheap)
# Target stage-1 sub-row length (elements). The reshape (M*C, N/C) puts the WHOLE
# sub-row of s = N/C elements into ONE row-kernel block's tile, so s trades off directly
# against occupancy: s at the smem ceiling (~44k fp32 = 175KB) -> ~1 block/SM -> ~3.5
# TB/s; s ~8-20k -> several blocks/SM -> ~7.3 TB/s on B200 (measured, >aten's 5.8).
# Below ~4k the stage-2 combine (one block folding C = N/s partials) starts to cost.
# 8192 sits in the flat top of that curve for both throughput and the ~0.6s compile.
_SUBROW_TARGET = 8192
# Stage-2 threads-per-block: the combine is bandwidth-light (folds C partials), so a
# mid block is plenty; parameterized so the autotuner / a retune can override.
_DEFAULT_BLOCK = 256


class _XctaConfig(NamedTuple):
    # xcta's own knob set; knobs left None are filled from _choose_config. subrow_target is the
    # stage-1 sub-row length TARGET, which _split_C snaps to a legal C by nearest-divisor search.
    block: int = _DEFAULT_BLOCK
    subrow_target: int = _SUBROW_TARGET


def _choose_config(hw=None, nfields: int = 1) -> "_XctaConfig":
    # xcta config. subrow_target is a stage-1 smem-tile-length sweet spot (elements), so
    # it scales with per-SM smem capacity: multiply the B200 anchor (8192) by
    # hw.smem_scale so a device with more/less smem targets a proportionally longer/
    # shorter sub-row (1.0 on B200 -> 8192 exactly). _split_C then snaps it to N's
    # nearest legal divisor. nfields is accepted for signature parity + autotuner key but
    # not acted on: the sweep showed wide-accumulator traits prefer shorter sub-rows, but
    # the shift is shape-dependent and no closed-form rule captures it without regressing
    # the nf=1 sweet spot -> left to the autotuner (which measures per key).
    smem = 1.0 if hw is None else hw.smem_scale
    return _XctaConfig(subrow_target=int(round(_SUBROW_TARGET * smem)))


def _split_C(N, vec, smem_budget_elems, subrow_target=None):
    # Choose C = chunks per output row for the two-stage split. The reshape
    # (M*C, N/C) requires C | N exactly, i.e. the sub-row length s = N/C must be a
    # DIVISOR of N that (1) is a multiple of vec (the row kernel vectorizes the load) and
    # (2) fits its tile (s <= smem_budget_elems). We do NOT maximize s -- that pins ~1
    # block/SM and halves bandwidth. Instead aim for subrow_target (the measured
    # occupancy sweet spot, default _SUBROW_TARGET) and take the nearest divisor,
    # searching outward. The search is bounded by the smem budget (~12k candidates)
    # regardless of N or its factorization, which matters when the divisors near the target lie
    # tens of thousands apart (N=4.396e9: divisors bracket the smem ceiling at 87920
    # and 98125). subrow_target is the exposed knob (autotuner overrides it).
    target = _SUBROW_TARGET if subrow_target is None else subrow_target
    step = max(vec, 1)
    # Floor on sub-row length: below this, stage 1 barely reduces (a subrow of 1 for a
    # prime N degenerates to C=N partials + a no-op stage 1). Keeps genuinely awkward N
    # (prime / only tiny divisors) on the backup K0 grid-stride path instead.
    lo = max(step, 256)
    hi = min(smem_budget_elems, N)
    hi -= hi % step
    if hi < lo:
        return None
    tgt = min(max(target - (target % step), lo), hi)
    # Expand symmetrically from tgt; first divisor of N (that keeps C <= _C_MAX) wins.
    for d in range(0, hi - lo + step, step):
        for s in (tgt + d, tgt - d):
            # C == 1 is rejected: a "split" into one chunk IS the one-shot, and every caller
            # reaches here only after the one-shot was declined -- so returning it rebuilds the
            # very kernel that was rejected. It arises for a PRIME N, whose only in-window
            # divisor is N itself: (8, 65537) bf16 then folded a 65537-element sub-row at
            # vec=1 and took 74us against ATen's 6.0. Declining sends it to the ragged split.
            if lo <= s <= hi and N % s == 0 and 1 < N // s <= _C_MAX:
                return N // s
    return None


class FusedTwoStage:
    # BOTH stage launches in ONE @cute.jit region -> one cute.compile artifact, one
    # host-side fn() call. Two cuLaunchKernel still issue on the device (serialized
    # on the stream, stage 2 sees stage 1's writes), but the expensive Python/
    # framework dispatch + arg marshalling is paid ONCE instead of twice. Keeps the
    # GOOD two-stage kernels (full grid, no cluster cap) AND graph-capturability.
    #
    # s1 is a kernel_rowtile.RowTile (final=False); s2 a kernel_general.ReduceBlock
    # (from_partials). Both expose a bound @cute.kernel `.kernel`; we replicate their
    # __call__ launch bodies here back-to-back.
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        parts: list,
        mOuts: list,
        rvals: list,
        kvals: list,
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
        # --- stage 2 launch (mirrors ReduceBlock.__call__); reads `parts` ---
        # Stage-2 grid = one block per output row = mOuts[0].shape[0] (read live, so this
        # fused kernel serves any M with no recompile). s2 has a single kept dim, so
        # _decode_offset ignores the extent -- correct for any M. The geometry
        # (count=C, project divisor N, decode quads) arrives as RUNTIME args, so one
        # compiled fused kernel serves every N in the vec class, each with its own C.
        # limit is unused (flat_tail=False) but the kernel signature requires it;
        # in_base is always 0 here.
        s2 = self.s2
        s2.kernel(
            parts,
            mOuts,
            rvals,
            kvals,
            count,
            cutlass.Int64(0),
            cutlass.Int64(count),
            project_n,
        ).launch(grid=[mOuts[0].shape[0], 1, 1], block=[s2.block, 1, 1], stream=stream)


# Two-level cache, as in kernel_general: _PLAN holds compiled kernels keyed on the sub-row's vec
# class and stage-1 config; _GEOM memoizes per-(N, knobs) derivations including the pre-boxed runtime
# args (~6us of Int boxing) and None declines.
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
    # Two-output form (max.dim/min.dim/aminmax/var_mean): stage 2 projects BOTH fields
    # of the same combined accumulator, so the split costs nothing extra over nouts==1.
    # Returns a tuple of nouts results, or None if the split is declined.
    return _reduce_row_xcta(
        trait, trait_key, x, list(out_dtypes), 2, block, flatten, subrow_target
    )


def _reduce_row_xcta(
    trait, trait_key, x, out_dtypes, nouts, block, flatten, subrow_target
):
    # FUSED two-stage row reduction. x: (M, N) contiguous, or (N,) for reduce-all; returns a tuple of
    # (M,) results. flatten=True collapses the M==1 result to a 0-d scalar, which is what a 1-D
    # reduce-ALL wants and a (1, N) reduce-DIM does not.
    #
    # The single @cute.jit region (FusedTwoStage) is what turns the few-row/huge-N regime from ~0.6x
    # ATen with two separate launches into ~1.15-1.44x in plain eager, and it captures cleanly.
    if not (x.is_cuda and x.is_contiguous()):
        raise AssertionError(
            f"need a contiguous CUDA input, got {x.device} {x.stride()}"
        )
    if x.dim() == 1:
        x = x.view(1, -1)
    M, N = x.shape
    # Fill block/subrow_target from the config when the caller passed None (the override
    # path); explicit values (autotuner) pass straight through. The hw-scaled
    # subrow_target is resolved here so _split_C snaps THAT (not its own raw anchor) to a
    # legal divisor; on B200 smem_scale=1.0 -> the 8192 anchor, unchanged.
    from .._cutedsl import hw_caps as _hw

    cfg = _choose_config(_hw.caps(x.device))
    block = cfg.block if block is None else block
    subrow_target = cfg.subrow_target if subrow_target is None else subrow_target

    # DYNAMIC M: the plan depends on N, dtype and trait but NOT M, so one compiled kernel serves any
    # batch size; M only sizes the grid and scratch. Stage 1 casts its tile coordinate to Int64, so a
    # >2^31 element offset does not overflow.
    #
    # TWO cache levels: a given N's GEOMETRY (its C split, pre-boxed args, wrap params) per gkey,
    # and the COMPILED kernel on the sub-row's vec class plus the stage-1 config -- both stages take
    # their extents as runtime args, so distinct N in a vec class share one kernel. subrow_target is
    # in gkey because it changes C.
    out_dtypes = tuple(out_dtypes)
    gkey = (trait_key, x.dtype, out_dtypes, N, block, subrow_target, str(x.device))
    geom = _GEOM.get(gkey)
    if geom is None and gkey not in _GEOM:
        geom = _GEOM[gkey] = _build_geom(
            trait, trait_key, x, out_dtypes, nouts, M, N, block, subrow_target
        )
    if geom is None:  # memoized refusal (prime / poorly-factored N) -> K0
        return None
    C, s, wrap_in, fn, rvals, kvals, cnt, pn, s1nc, s1nw = geom

    # One fused launch. Scratch partials sized to THIS M*C (allocated per call since M
    # varies); input + partials + output all dynamic-M so the cached kernel serves any
    # M. Stage 1 folds sub-rows -> raw partials; stage 2 combines per row.
    sub = x.reshape(M * C, s)
    parts = [
        torch.empty(M * C, device=x.device, dtype=_PART_TORCH[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    outs = [torch.empty(M, device=x.device, dtype=d) for d in out_dtypes]
    fn(
        wrap_in(sub),
        [_L.cute_tensor_dynM(p, ndim=1) for p in parts],
        [_L.cute_tensor_dynM(o, ndim=1) for o in outs],
        rvals,
        kvals,
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
    # Derive the C split for this exact N, then compile (or reuse) the fused kernel
    # for its bucket. Returns None to decline (no legal divisor split), memoized by
    # the caller so the K0 fallback is also remembered.
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
    # INDEX traits are declined here: stage 1 would have to rebase its within-sub-row column
    # to the global one, which the reshape makes awkward (the chunk index is row % C). They
    # are served by _two_stage_row's ragged split instead, whose gidx_from="chunk" already
    # carries the absolute reduced index -- measured 1.29-3.17x of ATen. See kernel_general.
    if getattr(trait, "has_index", False):
        return None
    svec = math.gcd(s, 128 // (elsize * 8))  # the sub-row's OWN vec class

    # ONE stage-1 shape: the row kernel's fold is ROLLED, so the sub-row length is a runtime arg
    # and one compiled kernel serves the whole vec class. N is absent from the key for that reason.
    cfg = _rt.row_config(s, elsize * 8, trait.nfields)
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
        return _rt.RowTile(
            trait, _L.torch2cute[x.dtype], s, tpr, nt, nouts, False, unroll
        )

    def wrap_in(t):
        return _L.cute_tensor_dynMN(t, svec, align=align, read_only=True)

    def _build():
        s1 = _make_s1()
        # s2's geometry values (count/project_n/quads) are runtime launch args; the
        # object only contributes the STRUCTURAL cache_sig fields (block, nfields,
        # single red/kept pair). The seed geometry below is per-build but the
        # compiled kernel is geometry-agnostic.
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
        # Dynamic-M seed wrappers; the compiled fn then serves any M.
        sub0 = x.reshape(M * C, s)
        parts0 = [
            torch.empty(M * C, device=device, dtype=_PART_TORCH[trait.fdtypes[f]])
            for f in range(trait.nfields)
        ]
        cparts0 = [_L.cute_tensor_dynM(p, ndim=1) for p in parts0]
        outs0 = [torch.empty(M, device=device, dtype=d) for d in out_dtypes]
        return _L.compile_kernel(
            fop,
            wrap_in(sub0),
            cparts0,
            [_L.cute_tensor_dynM(o, ndim=1) for o in outs0],
            *_s2_args(C, M, N),
            *s1_counts,
            _L.stream(),
        )

    fn = cached_plan(_PLAN, pkey, _build, op=f"aten::{trait_key}")
    return (C, s, wrap_in, fn, *_s2_args(C, M, N), *s1_counts)


def _s2_args(C, M, N):
    # Pre-boxed stage-2 runtime args (memoized in _GEOM: boxing is ~us-scale).
    # The kept-pair quad carries the seed M, but a single-pair decode ignores the
    # extent (rem * stride), so the baked M is dead -- M stays fully dynamic.
    return (
        _RB._quads([(C, 1)]),
        _RB._quads([(M, C)]),
        Int32(C),
        Int64(N),
    )
