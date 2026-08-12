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
# the device. Index traits (argmax/min) are served too: stage 1 rebases the
# chunk-local column to the GLOBAL one (index_rebase at runtime on the bucket
# path, baked index_chunks on the exact path), so stage-2 combine needs no remap.

import math
from typing import NamedTuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Int32, Int64

import torch

from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan
from . import (  # safe: kernel_general imports us only lazily
    kernel_general as _RB,
    kernel_row as _row,
)


_PART_TORCH = {Float32: torch.float32, Int32: torch.int32, Int64: torch.int64}


_C_MAX = 1 << 22  # cap stage-2 partial count (its combine is a dynamic loop -> cheap)
# Target stage-1 sub-row length (elements). The reshape (M*C, N/C) puts the WHOLE
# sub-row of s = N/C elements into ONE K1 block's smem tile, so s trades off directly
# against occupancy: s at the smem ceiling (~44k fp32 = 175KB) -> ~1 block/SM -> ~3.5
# TB/s; s ~8-20k -> several blocks/SM -> ~7.3 TB/s on B200 (measured, >aten's 5.8).
# Below ~4k the stage-2 combine (one block folding C = N/s partials) starts to cost.
# 8192 sits in the flat top of that curve for both throughput and the ~0.6s compile.
_SUBROW_TARGET = 8192
# Stage-2 threads-per-block: the combine is bandwidth-light (folds C partials), so a
# mid block is plenty; parameterized so the autotuner / a retune can override.
_DEFAULT_BLOCK = 256


class _XctaConfig(NamedTuple):
    # xcta two-stage knob set. reduce_row_xcta() knobs left None are filled from
    # _choose_config; explicit values override per field. This is xcta's OWN config
    # (row/col/K0 have differently-shaped knobs). subrow_target is the stage-1 sub-row
    # length TARGET (a nearest-DIVISOR search in _split_C snaps it to a legal C -- the
    # search is algorithmic, not a table, because it depends on N's factorization).
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
        self,
        mX: cute.Tensor,
        parts: list,
        mOut: cute.Tensor,
        rvals: list,
        kvals: list,
        count: cutlass.Int32,
        project_n: cutlass.Int64,
        stream: cuda.CUstream,
    ):
        s1 = self.s1
        # --- stage 1 launch (mirrors RowReduce.__call__) ---
        # s1.vec == gcd(N, 128//width) on the exact path and the bucket's vec class
        # in bucket mode (dyn_n_vec) -- RowReduce.__init__ resolves both. count (=C)
        # doubles as the index_rebase chunk count when s1 carries an index trait.
        s1._set_cluster_n()
        tiled_copy, tiler_mn, threads_per_row = s1._get_tiled_copy(
            vecsize=const_expr(s1.vec)
        )
        num_threads = tiled_copy.size
        s1.kernel(mX, parts, tiler_mn, tiled_copy, threads_per_row, None, count).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )
        # --- stage 2 launch (mirrors ReduceBlock.__call__); reads `parts` ---
        # Stage-2 grid = one block per output row = mOut.shape[0] (read live, so this
        # fused kernel serves any M with no recompile). s2 has a single kept dim, so
        # _decode_offset ignores the extent -- correct for any M. The geometry
        # (count=C, project divisor N, decode quads) arrives as RUNTIME args -- in
        # bucket mode one compiled fused kernel serves every N in the sub-row-ceiling
        # bucket, each with its own C. limit is unused (flat_tail=False) but the
        # kernel signature requires it; in_base is always 0 here.
        s2 = self.s2
        s2.kernel(
            parts,
            [mOut],
            rvals,
            kvals,
            count,
            cutlass.Int64(0),
            cutlass.Int64(count),
            project_n,
        ).launch(grid=[mOut.shape[0], 1, 1], block=[s2.block, 1, 1], stream=stream)


_DEV_SM = {}  # device -> multi_processor_count (get_device_properties is ~1.3us)
# Two-level cache (mirrors kernel_general): _PLAN holds COMPILED fused kernels --
# keyed on the sub-row-ceiling BUCKET (one kernel serves every N whose snapped
# sub-row falls in the bucket) or exact geometry for the index-trait/knob paths.
# _GEOM memoizes per-(N, knobs) derivations: C, the wrap params, and the PRE-BOXED
# runtime args (quads + Int boxing is ~6us), including None declines (prime N).
_PLAN = {}
_GEOM = {}


def _device_sm(device):
    sm = _DEV_SM.get(device)
    if sm is None:
        sm = torch.cuda.get_device_properties(device).multi_processor_count
        _DEV_SM[device] = sm
    return sm


def reduce_row_xcta(
    trait, trait_key, x, out_dtype, block=None, flatten=False, subrow_target=None
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
    # Fill block/subrow_target from the config when the caller passed None (the override
    # path); explicit values (autotuner) pass straight through. The hw-scaled
    # subrow_target is resolved here so _split_C snaps THAT (not its own raw anchor) to a
    # legal divisor; on B200 smem_scale=1.0 -> the 8192 anchor, unchanged.
    from .._cutedsl import hw_caps as _hw

    cfg = _choose_config(_hw.caps(x.device))
    block = cfg.block if block is None else block
    subrow_target = cfg.subrow_target if subrow_target is None else subrow_target

    # DYNAMIC M: the plan (C, ops, compiled fn) depends only on N+dtype+trait, NOT M.
    # One compiled fused kernel serves any M (e.g. every batch size in a training
    # loop) with no recompile. M only sizes the grid + scratch, both handled live.
    # (Stage-1 K1 casts its block-index tile coordinate to Int64, so the >2^31 M*N
    # offset overflow that previously broke large-M is fixed -- see kernel_row.)
    #
    # TWO cache levels (recompile minimization): the GEOMETRY of a given N (its C
    # split, pre-boxed runtime args, wrap params) is memoized per gkey; the COMPILED
    # fused kernel is keyed on the sub-row's (vec-class, bucket-ceiling) BUCKET --
    # stage 1 runs in RowReduce bucket mode (dyn_n_vec: static ceiling tile,
    # column-predicated loads, runtime sub-row length) and stage 2 takes C/N as
    # runtime args, so distinct N whose snapped sub-rows share a bucket share ONE
    # kernel. Index traits keep an exact-N kernel: their stage-1 global-column
    # rebase (index_chunks) bakes C. subrow_target is in gkey: it changes C.
    gkey = (trait_key, x.dtype, out_dtype, N, block, subrow_target, str(x.device))
    geom = _GEOM.get(gkey)
    if geom is None and gkey not in _GEOM:
        geom = _GEOM[gkey] = _build_geom(
            trait, trait_key, x, out_dtype, M, N, block, subrow_target
        )
    if geom is None:  # memoized refusal (prime / poorly-factored N) -> K0
        return None
    C, s, wrap_in, fn, rvals, kvals, cnt, pn = geom

    # One fused launch. Scratch partials sized to THIS M*C (allocated per call since M
    # varies); input + partials + output all dynamic-M so the cached kernel serves any
    # M. Stage 1 folds sub-rows -> raw partials; stage 2 combines per row.
    sub = x.reshape(M * C, s)
    parts = [
        torch.empty(M * C, device=x.device, dtype=_PART_TORCH[trait.fdtypes[f]])
        for f in range(trait.nfields)
    ]
    out = torch.empty(M, device=x.device, dtype=out_dtype)
    fn(
        wrap_in(sub),
        [_L.cute_tensor_dynM(p, ndim=1) for p in parts],
        _L.cute_tensor_dynM(out, ndim=1),
        rvals,
        kvals,
        cnt,
        pn,
        _L.stream(),
    )
    return out.view(()) if flatten and out.numel() == 1 and M == 1 else out


def _build_geom(trait, trait_key, x, out_dtype, M, N, block, subrow_target):
    # Derive the C split for this exact N, then compile (or reuse) the fused kernel
    # for its bucket. Returns None to decline (no legal divisor split), memoized by
    # the caller so the K0 fallback is also remembered.
    device = x.device
    elsize = x.element_size()
    sm = _device_sm(device)
    vec = math.gcd(N, 128 // (elsize * 8))
    # M only sizes the device-fill heuristic in _split_C; C is fixed in the plan
    # (it sets stage-1's sub-row length N//C and stage-2's partial count), so the
    # same C must be used for every M -> derive it once here.
    C = _split_C(M, N, vec, sm, _RB._SMEM_BUDGET // elsize, subrow_target)
    if C is None:
        # Prime / poorly-factored N: no clean reshape split. The caller memoizes the
        # None -> the K0 general kernel serves it (any N, no reshape, O(1) compile).
        return None
    s = N // C
    has_index = getattr(trait, "has_index", False)
    svec = math.gcd(s, 128 // (elsize * 8))  # the sub-row's OWN vec class
    ceiling = _row._bucket_ceiling(s, elsize)

    if ceiling is not None:
        # BUCKET stage 1: one compiled fused kernel per (vec-class, ceiling) serves
        # every N whose snapped sub-row lands in the bucket. Index traits are
        # served too: index_rebase computes the global-column base from the
        # RUNTIME (C, sub-row length) -- C rides FusedTwoStage's `count` arg and s
        # is the input's dynamic column extent -- so nothing N-derived is baked.
        pkey = (
            "xctab",
            trait_key,
            x.dtype,
            out_dtype,
            svec,
            ceiling,
            block,
            str(device),
        )
        align = svec * elsize

        def _make_s1():
            return _row.RowReduce(
                trait,
                _L.torch2cute[x.dtype],
                ceiling,
                final=False,
                dyn_n_vec=svec,
                index_rebase=has_index,
            )

        def wrap_in(t):
            return _L.cute_tensor_dynMN(t, svec, align=align, read_only=True)

    else:
        # EXACT stage 1: a sub-row whose bucket rung would blow the smem budget.
        # (Index traits use the baked index_chunks=C here -- mirrors ATen carrying
        # the absolute index so stage-2 combine needs no remap.)
        pkey = ("xcta", trait_key, x.dtype, out_dtype, N, block, s, str(device))

        def _make_s1():
            return _row.RowReduce(
                trait, _L.torch2cute[x.dtype], s, final=False, index_chunks=C
            )

        s1_for_align = _make_s1()

        def wrap_in(t):
            return _row._aligned_in_dynM(t, s1_for_align)

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
            nouts=1,
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
        out0 = torch.empty(M, device=device, dtype=out_dtype)
        return _L.compile(
            fop,
            wrap_in(sub0),
            cparts0,
            _L.cute_tensor_dynM(out0, ndim=1),
            *_s2_args(C, M, N),
            _L.stream(),
        )

    fn = cached_plan(_PLAN, pkey, _build)
    return (C, s, wrap_in, fn, *_s2_args(C, M, N))


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
