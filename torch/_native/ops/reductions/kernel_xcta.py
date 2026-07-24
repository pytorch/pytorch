# TWO-STAGE cross-CTA row reduction for few-row / huge-N reductions (and the M=1
# reduce-all case), mirroring ATen's Reduce.cuh structure.
#
# An earlier design used a SINGLE kernel: grid (C, M), every block reduced a chunk
# then a semaphore elected one "last" block to combine the C partials. That made
# ALL C*M blocks run a GPU-scope fence + a redundant finalize after phase 1; for a
# huge single row (C ~ 1024) those serialized down to ~36% DRAM (confirmed by ncu:
# 36% dram, 11% sm, 77% warp occupancy -- latency/sync bound, not bandwidth bound).
# The clean two-stage split removes ALL cross-block sync from the bandwidth-bound
# pass and beats both the semaphore design and ATen (5.4 TB/s on (1,16M) fp32 =
# 1.13x ATen; fp16 up to 1.66x).
#
#   stage 1: reshape (M, N) -> (M*C, N/C). The vectorized K1 row kernel
#            (kernel_row) reduces each sub-row to a RAW per-field accumulator
#            (final=False -> no projection). No cross-block sync -> full bandwidth.
#   stage 2: ReduceBlock from_partials COMBINES the C partials of each row (one
#            block per row) and projects ONCE with the true N. Using combine (not
#            re-reduce) keeps mean/var/norm correct, not just sum.
#
# C is chosen so each stage-1 sub-row fits K1's smem tile AND the M*C blocks fill
# the device. Index traits (argmax/min) do NOT come here -- the dispatcher keeps
# them on the general kernel (their stage-2 needs a global-index remap not built).

import math

import cuda.bindings.driver as cuda

import cutlass.cute as cute
from cutlass import const_expr, Float32, Float64, Int32, Int64

import torch

from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan
from . import (  # safe: kernel_general imports us only lazily
    kernel_general as _RB,
    kernel_row as _row,
)


_PART_TORCH = {
    Float32: torch.float32,
    Float64: torch.float64,
    Int32: torch.int32,
    Int64: torch.int64,
}


_C_MAX = 1 << 22  # cap stage-2 partial count (its combine is a dynamic loop -> cheap)
# Target stage-1 sub-row length (elements). The reshape (M*C, N/C) puts the WHOLE
# sub-row of s = N/C elements into ONE K1 block's smem tile, so s trades off directly
# against occupancy: s at the smem ceiling (~44k fp32 = 175KB) -> ~1 block/SM -> ~3.5
# TB/s; s ~8-20k -> several blocks/SM -> ~7.3 TB/s on B200 (measured, >aten's 5.8).
# Below ~4k the stage-2 combine (one block folding C = N/s partials) starts to cost.
# 8192 sits in the flat top of that curve for both throughput and the ~0.6s compile.
_SUBROW_TARGET = 8192


def _split_C(M, N, vec, sm, smem_budget_elems, subrow_target=None):
    # Choose C = chunks per output row for the two-stage split. The reshape
    # (M*C, N/C) requires C | N exactly, i.e. the sub-row length s = N/C must be a
    # DIVISOR of N that (1) is a multiple of vec (K1 vectorizes the load) and (2) fits
    # K1's smem tile (s <= smem_budget_elems). We do NOT maximize s -- that pins ~1
    # block/SM and halves bandwidth. Instead aim for subrow_target (the measured
    # occupancy sweet spot, default _SUBROW_TARGET) and take the nearest divisor,
    # searching outward. The search is bounded by the smem budget (~12k candidates)
    # regardless of N or its factorization -- unlike the old width-1 window around a
    # single C target, which for a huge N missed divisors tens of thousands apart and
    # fell to the slow backup (N=4.396e9: divisors bracket the smem ceiling at 87920
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
            if lo <= s <= hi and N % s == 0 and N // s <= _C_MAX:
                return N // s
    return None


class FusedTwoStage:
    # BOTH stage launches in ONE @cute.jit region -> one cute.compile artifact, one
    # host-side fn() call. Two cuLaunchKernel still issue on the device (serialized
    # on the stream, stage 2 sees stage 1's writes), but the expensive Python/
    # framework dispatch + arg marshalling is paid ONCE instead of twice. Keeps the
    # GOOD two-stage kernels (full grid, no cluster cap) AND graph-capturability.
    #
    # s1 is a kernel_row.RowReduce (final=False); s2 a kernel_general.ReduceBlock
    # (from_partials). Both expose a bound @cute.kernel `.kernel`; we replicate their
    # __call__ launch bodies here back-to-back.
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

    @cute.jit
    def __call__(
        self, mX: cute.Tensor, parts: list, mOut: cute.Tensor, stream: cuda.CUstream
    ):
        s1 = self.s1
        # --- stage 1 launch (mirrors RowReduce.__call__) ---
        s1._set_cluster_n()
        vecsize = const_expr(math.gcd(s1.N, 128 // s1.dtype.width))
        tiled_copy, tiler_mn, threads_per_row = s1._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        s1.kernel(mX, parts, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )
        # --- stage 2 launch (mirrors ReduceBlock.__call__); reads `parts` ---
        # Stage-2 grid = one block per output row = mOut.shape[0] (read live, so this
        # fused kernel serves any M with no recompile). s2 has a single kept dim, so
        # _decode_offset ignores the extent -- correct for any M.
        s2 = self.s2
        s2.kernel(parts, [mOut]).launch(
            grid=[mOut.shape[0], 1, 1], block=[s2.block, 1, 1], stream=stream
        )


_DEV_SM = {}  # device -> multi_processor_count (get_device_properties is ~1.3us)
_PLAN = {}  # shape-key -> derived launch plan (everything shape-invariant)


def _device_sm(device):
    sm = _DEV_SM.get(device)
    if sm is None:
        sm = torch.cuda.get_device_properties(device).multi_processor_count
        _DEV_SM[device] = sm
    return sm


def reduce_row_xcta(
    trait, trait_key, x, out_dtype, block=256, flatten=False, subrow_target=None
):
    # FUSED two-stage row reduction (reduce last dim). x: (M, N) contiguous, or (N,)
    # for reduce-all. Returns (M,) results. flatten=True (reduce-ALL of a 1-D input)
    # collapses the single (M==1) result to a 0-d scalar; a 2-D (1, N) reduce-DIM
    # caller leaves flatten=False so the result stays (1,) -- matching aten's shape.
    #
    # Both stage launches live in ONE @cute.jit region (FusedTwoStage) -> ONE
    # cute.compile artifact + ONE host-side fn() call. The two cuLaunchKernel still
    # issue on the device (serialized on the stream, stage 2 reads stage 1's
    # partials), but the expensive Python/framework dispatch is paid ONCE instead of
    # twice. This is what turns the few-row/huge-N regime from ~0.6x ATen (two
    # separate launches) into ~1.15-1.44x in plain eager -- graph-class perf without
    # graphs. It also captures into CUDA graphs cleanly.
    assert x.is_cuda and x.is_contiguous()  # noqa: S101
    if x.dim() == 1:
        x = x.view(1, -1)
    M, N = x.shape

    # DYNAMIC M: the plan (C, ops, compiled fn) depends only on N+dtype+trait, NOT M.
    # One compiled fused kernel serves any M (e.g. every batch size in a training
    # loop) with no recompile. M only sizes the grid + scratch, both handled live.
    # (Stage-1 K1 casts its block-index tile coordinate to Int64, so the >2^31 M*N
    # offset overflow that previously broke large-M is fixed -- see kernel_row.)
    # subrow_target is in the key: it changes C (the reshape split), so distinct
    # values compile distinct plans. None -> the _SUBROW_TARGET heuristic.
    pkey = (trait_key, x.dtype, out_dtype, N, block, subrow_target, str(x.device))

    def _build():
        sm = _device_sm(x.device)
        vec = math.gcd(N, 128 // (x.element_size() * 8))
        # M only sizes the device-fill heuristic in _split_C; C is fixed in the plan
        # (it sets stage-1's sub-row length N//C and stage-2's partial count), so the
        # same C must be used for every M -> derive it once here.
        C = _split_C(M, N, vec, sm, _RB._SMEM_BUDGET // x.element_size(), subrow_target)
        if C is None:
            # Prime / poorly-factored N: no clean reshape split. Plan is None ->
            # caller uses the K0 general kernel (any N, no reshape, O(1) compile).
            return None
        # index_chunks=C so an INDEX trait's stage-1 fold accumulates the GLOBAL
        # column (chunk-local + chunk base), making the partial index absolute -- so
        # stage-2 combine needs no remap (mirrors ATen carrying the absolute index).
        # For non-index traits C is harmless (the index arg is ignored).
        s1 = _row.RowReduce(
            trait, _L.torch2cute[x.dtype], N // C, final=False, index_chunks=C
        )
        s2 = _RB.ReduceBlock(
            trait,
            count=C,
            num_o=M,
            red_pairs=[(C, 1)],
            kept_pairs=[(M, C)],
            from_partials=True,
            project_n=N,
            nouts=1,
            final=True,
            block=block,
            dyn_num_o=True,
        )
        fop = FusedTwoStage(s1, s2)
        # Dynamic-M seed wrappers; the compiled fn then serves any M.
        sub0 = x.reshape(M * C, N // C)
        parts0 = [
            torch.empty(M * C, device=x.device, dtype=_PART_TORCH[trait.fdtypes[f]])
            for f in range(trait.nfields)
        ]
        cparts0 = [_L.cute_tensor_dynM(p, ndim=1) for p in parts0]
        out0 = torch.empty(M, device=x.device, dtype=out_dtype)
        fn = _L.compile(
            fop,
            _row._aligned_in_dynM(sub0, s1),
            cparts0,
            _L.cute_tensor_dynM(out0, ndim=1),
            _L.stream(),
        )
        return (C, s1, fn)

    plan = cached_plan(_PLAN, pkey, _build)
    if plan is None:  # memoized refusal (prime N) -> caller uses K0
        return None
    C, s1, fn = plan

    # One fused launch. Scratch partials sized to THIS M*C (allocated per call since M
    # varies); input + partials + output all dynamic-M so the cached kernel serves any
    # M. Stage 1 folds sub-rows -> raw partials; stage 2 combines per row.
    sub = x.reshape(M * C, N // C)
    parts = [
        torch.empty(M * C, device=x.device, dtype=_PART_TORCH[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    out = torch.empty(M, device=x.device, dtype=out_dtype)
    fn(
        _row._aligned_in_dynM(sub, s1),
        [_L.cute_tensor_dynM(p, ndim=1) for p in parts],
        _L.cute_tensor_dynM(out, ndim=1),
        _L.stream(),
    )
    return out.view(()) if flatten and out.numel() == 1 and M == 1 else out
