# Fragment-based unified row reduction, built on quack's ReductionBase.
#
# WHY ReductionBase: the v1 kernel folded with a CONSTEXPR loop over count/block
# elements, so unroll depth scaled with N -- a 16M-element row wants a 131072-deep
# unroll and never compiles. v1 also did scalar loads. ReductionBase provides the
# proven vectorized-tiled-load scaffolding (TiledCopy + cp.async + smem) that the
# production rmsnorm/softmax kernels use; we reuse it verbatim and only swap in
# OUR trait fold + per-field cross-thread reduce.
#
# HOW the trait survives the DSL: quack's own grid-stride loops only ever call
# DSL enums / module funcs (operator.add, cute.arch.fmax) -- never a user-class
# bound method, which is what trips the IR flattener ("'for' encountered a
# user-defined Python object"). So we do NOT put trait calls in any dynamic loop.
# Instead the whole row tile is loaded into a register fragment (N is a
# compile-time const -> fixed fragment size), and the per-thread fold over that
# fragment is a CONSTEXPR loop. Trait calls in a constexpr loop are fine (this is
# exactly what v1 did and what the trait library was validated on). The dynamic
# work (the vectorized copy) is pure DSL ops with no trait reference.
#
# Trait protocol + warp_reduce/block_reduce reused unchanged from reduce_traits.
# Row / last-dim-contiguous geometry only -- the perf-critical common case.
#
# STATUS: validated (full 2018-case suite passes with this wired in as the row
# fast path via kernel_general._try_fast_row) and memory-clean (0 compute-sanitizer
# errors on ragged-N). fp32 row sum hits ~6.6 TB/s on (8192,8192) = ~1.5x torch
# and ~80% of B200 peak. The old small-M/wide-N fp16 loss (2048x16384 ~2x behind
# torch) was fixed by the _WIDE_ROW_BYTES tpr256 rung -- remeasured 1.4-2x AHEAD
# across the wide-row class (bf16/fp16/fp32). CAVEAT: the whole row goes into ONE
# block's register fragment, so compile time and register pressure grow with N --
# capped at N<=65536 by kernel_general._MAX_N, above which the general kernel
# handles it.

import math
from typing import NamedTuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, pipeline
from cutlass.cute.nvgpu import cpasync

import torch
from torch._vendor.quack import copy_utils
from torch._vendor.quack.reduction_base import ReductionBase

from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan
from .._cutedsl.traits import block_reduce, cluster_reduce, WARP, warp_reduce


_cute = _L.cute_tensor
_compile = _L.compile  # cute.compile + options="--enable-tvm-ffi"
_stream = _L.stream


# --- K1 row-reduce occupancy heuristic, parameterized as DATA (see _choose_config) ---
# threads-per-row (tpr) ladder: the first (eff_N <= limit) row wins; above the last,
# TPR_MAX. From quack's production rmsnorm tuning (rmsnorm_config_for_hopper_fwd): small
# N -> 1 warp/row so many rows pack per block with no cross-warp reduce. Autotuner
# overrides. The N-limits are B200-tuned anchors; _choose_config scales them by hw so the
# same ladder generalizes across GPUs (identity on B200).
_TPR_LADDER = ((64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128))
_TPR_MAX = 256
# threads-per-block (nt) gate: small rows use a smaller block, wider rows the larger.
_NT_SMALL, _NT_LARGE, _NT_GATE_N = 128, 256, 16 * 1024
# Wide-row fast rung: once a ROW is >= this many BYTES it needs the full 256 threads/row
# (+256/block) -- the element ladder above is dtype-blind and under-threads here. Measured
# (graph-mode K1 sweep): tpr256/nt256 wins from row_bytes >= 16KB for BOTH fp32 (N>=4096)
# and bf16 (N>=8192) -- a clean byte-aligned crossover, 1.1-1.4x vs the ladder's tpr64/128
# -- and loses badly below it (small N wants many rows/block). Bytes, not elements, so it
# is dtype-correct without a per-dtype table.
_WIDE_ROW_BYTES = 16 * 1024


class _RowConfig(NamedTuple):
    # K1 row-reduce knob set. reduce_row() knobs left None are filled from _choose_config;
    # explicit values (autotuner / per-machine retune) override per field. This is K1's
    # OWN config (col/xcta/K0 have differently-shaped knobs).
    tpr: int  # threads per row
    nt: int  # threads per block


def _choose_config(N: int, dtype_width: int, nfields: int = 1, hw=None) -> "_RowConfig":
    # Occupancy config from (N, dtype), with hw-scaled thresholds. tpr from the ladder,
    # nt from the size gate; the ladder's N-limits are proxies for "how wide before a row
    # needs more threads", which tracks smem capacity, so scaling them by hw.smem_scale
    # keeps the rule correct on other GPUs (1.0 on B200 -> the anchors reproduce exactly).
    #
    # nfields is ACCEPTED (signature parity + autotuner key) but deliberately NOT acted
    # on. The nfields sweep confirmed wide-accumulator traits (var nf=3) prefer a
    # different config, BUT the fp32 and bf16 optima move in OPPOSITE directions (fp32
    # var wants larger tpr / smaller nt; bf16 var the reverse), so no single scalar
    # rule serves both -- a measured `eff = N*nfields` fit REGRESSED bf16 var to ~0.92x
    # while helping fp32. Until a per-(dtype, nfields) table is characterized densely
    # enough to interpolate, nfields is left to the autotuner (which measures per key and
    # cannot regress). nf=1 (sum/mean/prod/norm/...) is unaffected and keeps its picks.
    # The ragged-N correctness clamp (tpr -> nt when the row tile isn't a clean vec*tpr
    # multiple) is applied at USE in RowReduce, not here -- it is a correctness invariant.
    smem = 1.0 if hw is None else hw.smem_scale
    # Wide-row rung first (byte-based, dtype-correct): a >=16KB row saturates 256
    # threads/row + a 256 block regardless of dtype. This overrides the element ladder,
    # which under-threads mid-N wide rows (e.g. fp32 N=4096-8192 -> tpr64/128, ~1.3x slow).
    if N * (dtype_width // 8) >= _WIDE_ROW_BYTES * smem:
        return _RowConfig(tpr=_TPR_MAX, nt=_NT_LARGE)
    tpr = next((t for limit, t in _TPR_LADDER if N <= limit * smem), _TPR_MAX)
    nt = _NT_SMALL if N <= _NT_GATE_N * smem else _NT_LARGE
    return _RowConfig(tpr=tpr, nt=nt)


class RowReduce(ReductionBase):
    # Reduce the contiguous last dim of (M, N) -> (M,). One CTA handles
    # tiler_mn[0] rows; each thread vector-loads its slice of a row, folds the
    # fragment with the trait, then warp+block reduce across the row.
    #
    # `final` (default True) projects the accumulator and writes one result per
    # row. `final=False` writes the RAW per-field accumulators to nfields partial
    # buffers instead -- this is stage 1 of the multi-CTA-per-row split, where a
    # long row is cut into chunks (each handled as its own "row" of a reshaped
    # (M*C, N/C) input) and a cheap stage-2 row reduction combines the C partials
    # and projects once with the TRUE row length (project_n).
    def __init__(
        self,
        trait,
        dtype,
        N,
        final=True,
        project_n=None,
        nouts=1,
        index_chunks=1,
        tpr_override=None,
        nt_override=None,
        cp_async=True,
        cluster_n=1,
        use_tma=False,
        dyn_n_vec=None,
        index_rebase=False,
    ):
        # stage=1: a single reduction buffer (we drive the combine ourselves).
        super().__init__(dtype, N, stage=1)
        self.trait = trait
        self.final = final
        # dyn_n_vec: BUCKET mode (recompile minimization). N is then a STATIC tile
        # CEILING (>= every served row length), and the true row length is read from
        # mX.shape[1] at runtime: the load is column-predicated (the quack rmsnorm
        # ragged idiom), the fold guards col < N, and the project divisor is the
        # runtime N. One compiled kernel serves EVERY N in a (vec-class, ceiling)
        # bucket instead of one per exact N. The value is the bucket's vector class
        # (divides every served N; keeps the wide-load width). Excludes cluster/TMA
        # (both need static extents) and the xcta stage-1 index-chunk split.
        self.dyn_n_vec = dyn_n_vec
        self.nouts = nouts  # 1 (most ops) or 2 (max.dim/min.dim: value + index).
        # Exposed occupancy knobs (autotuner overrides; None -> the tuned ladder):
        #   tpr_override: threads PER ROW, replacing the _threads_per_row ladder pick.
        #   nt_override:  threads per BLOCK, replacing the _num_threads 128/256 gate.
        # The ragged-N correctness guard in _threads_per_row still applies to the
        # override (a multi-row tile needs N a clean multiple of vec*tpr).
        self.tpr_override = tpr_override
        self.nt_override = nt_override
        # cluster_n: split each row across cluster_n CTAs (a launch cluster); each CTA
        # reduces a distinct N/cluster_n column slice, then a cross-CTA cluster reduce
        # (traits.cluster_reduce, via distributed-shared-memory mbarrier) folds the
        # peers. 1 = no clustering (the default; every cluster branch below is a
        # compile-time no-op, so the kernel is identical to the pre-cluster version).
        # Requires sm_90+; the host builder caps it (see reduce_row_cluster).
        self.cluster_n = cluster_n
        # cp_async: use the SMEM-staged wide cp.async load path when its preconditions
        # (even N, vector >= 32 bits) hold. False forces the direct gmem->rmem path
        # even when cp.async is legal -- an exposed knob, since skipping smem staging
        # can win on occupancy for some shapes. It NEVER enables cp.async where it is
        # illegal (ragged/narrow): that gate stays a hard correctness requirement.
        self.cp_async = cp_async
        # use_tma: load the (tiler_m, N) tile gmem->smem via the TMA unit
        # (cp.async.bulk.tensor) instead of cp.async/direct. A third load path filling
        # the SAME staged sX tile the fragment fold reads; the reduce math is
        # unchanged. Driven by the tested PipelineTmaAsync (correct mbarrier count /
        # fence / phase -- hand-rolling deadlocks). Requires an even tile (TMA moves
        # the whole static box) and is incompatible with cluster_n>1 and the
        # index-chunk path; the host builder only enables it in the plain one-shot
        # row-sum-class case (see reduce_row use_tma gate).
        self.use_tma = use_tma
        # index_chunks (C): for INDEX traits used as the xcta two-stage stage-1, the
        # input (M, N) is reshaped to (M*C, N/C), so each sub-row's column index is
        # CHUNK-LOCAL [0, N/C). To make the accumulated index the true GLOBAL column
        # [0, N), the fold adds (sub_row % C) * (N/C) -- the chunk's column base. C=1
        # (the default, non-split case) makes this a no-op. Mirrors ATen carrying the
        # absolute index in the partial so stage-2 combine needs no remap.
        self.index_chunks = index_chunks
        # index_rebase: RUNTIME variant of index_chunks for BUCKET mode. Both
        # rebase an index trait's chunk-local column to the GLOBAL one after the
        # xcta (M, N) -> (M*C, s) reshape: col_base = (sub_row % C) * s.
        # index_chunks bakes C and s as const_expr (exact-N kernels);
        # index_rebase reads BOTH at runtime -- C as the c_chunks kernel arg, s as
        # the input's dynamic shape[1] -- so one bucket kernel serves every
        # (N, C) whose sub-row lands in the ceiling. The mod runs once per thread
        # before the fold (not in the hot loop). Exclusive with index_chunks>1.
        self.index_rebase = index_rebase
        self.project_n = project_n if project_n is not None else N
        if dyn_n_vec is not None:
            # Bucket mode is one-shot only: cluster/TMA bake static extents; the
            # baked index_chunks path needs the exact N (use index_rebase instead).
            assert cluster_n == 1 and not use_tma and index_chunks == 1  # noqa: S101
            self.vec = dyn_n_vec  # the bucket's vector class, not gcd of the ceiling
        else:
            self.vec = math.gcd(N, 128 // dtype.width)  # elems per 128-bit vector

    def _num_threads(self):
        if self.nt_override is not None:
            return self.nt_override
        return _choose_config(self.N, self.dtype.width, self.trait.nfields).nt

    def _threads_per_row(self):
        # CRITICAL for occupancy: use only as many threads PER ROW as the row needs, so
        # num_threads/threads_per_row rows pack into each block (see _choose_config /
        # _TPR_LADDER for the parameterized pick). Overriding maxes this out (1 row/
        # block), starving occupancy and forcing a cross-warp smem reduce for tiny rows.
        # nfields scales the effective width so wide-accumulator traits (var nf=3) pick
        # a config matching their higher per-lane pressure.
        nt = self._num_threads()
        tpr = (
            self.tpr_override
            if self.tpr_override is not None
            else _choose_config(self.N, self.dtype.width, self.trait.nfields).tpr
        )
        # Packing >1 row per block (nt//tpr) is only correct when each row's tile
        # is exactly N wide. For RAGGED N the tile is padded to a multiple of
        # vec*tpr, so a multi-row tile's row stride != N and rows misalign. Force
        # 1 row/block (tpr == num_threads) when N isn't a clean multiple. Even N
        # keeps the occupancy-friendly small tpr.
        vec = math.gcd(self.N, 128 // self.dtype.width)
        even = self.N % (vec * tpr) == 0
        tpr = tpr if even else nt
        # Floor at ONE WARP. The ladder's small-N rungs give tpr 8/16, but the
        # cross-thread reduce below divides by warps_per_row = tpr // WARP, which
        # floors to 0 for a sub-warp tpr -> ZeroDivisionError at trace time (a hard
        # user-facing crash, not a fallback: x.sum(dim=1) died for N=32/64/128).
        # A sub-warp tpr cannot work anyway -- warp_reduce shuffles across a full
        # warp, so a partial warp would fold lanes belonging to other rows.
        return max(tpr, WARP)

    @cute.jit
    def __call__(self, mX: cute.Tensor, mOuts: list, stream: cuda.CUstream):
        # mOuts: [result] when final; [partial_field_0, ...] when not final.
        # cluster_n is set by the host builder (const), NOT _set_cluster_n() (which
        # would reset it to 1); _get_tiled_copy reads it to narrow each CTA's column
        # tile to N/cluster_n. grid.y = cluster_n places the peer CTAs in one cluster.
        # Bucket mode: self.vec IS the bucket's vector class (set in __init__); the
        # static form is gcd of the exact N.
        vecsize = const_expr(self.vec)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        # TMA atom (host-compile-time, inside the jit region). mX_tma replaces mX for
        # the load; the atom is baked into the kernel. Only when use_tma is on.
        if const_expr(self.use_tma):
            tma_smem_layout = cute.make_ordered_layout(tiler_mn, order=(1, 0))
            tma_atom, mX_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), mX, tma_smem_layout, tiler_mn
            )
        else:
            tma_atom, mX_tma = None, mX
        self.kernel(
            mX_tma, mOuts, tiler_mn, tiled_copy, threads_per_row, tma_atom
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=(
                [1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None
            ),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mOuts: list,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr,
        tma_atom=None,
        c_chunks=None,
    ):
        # c_chunks: RUNTIME chunk count for index_rebase (bucket-mode xcta stage 1
        # with an index trait); None otherwise. Only kernel_xcta's FusedTwoStage
        # passes it (it already carries C as a runtime launch arg).
        trait = self.trait
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        # Cluster column tile: CTA rank `cluster_y` in the cluster owns column-tile
        # `cluster_y` of the row (tiler_mn[1] = N/cluster_n wide). cluster_n==1 ->
        # cluster_y is the compile-time 0 -> identical to the single-CTA tiling.
        # block_idx_in_cluster() returns the LINEARIZED rank; our cluster is
        # [1, cluster_n, 1], so the linear rank IS the y-rank (0..cluster_n-1).
        cluster_y = (
            const_expr(0)
            if const_expr(self.cluster_n == 1)
            else cute.arch.block_idx_in_cluster()
        )

        shape = (cute.size(mX, mode=[0]), cute.size(mX, mode=[1]))
        # INT64 tile coordinate: local_tile computes the tile's gmem offset in the
        # COORDINATE's integer type. block_idx() is i32, so (bidx, 0) overflows the
        # offset when M*N >= 2^31 (e.g. 300000x8192) -> negative wrap -> illegal
        # access. Cast the block index to Int64 so the offset math is 64-bit. (cX
        # indexes the small identity tensor so it can't overflow, but keep it
        # consistent.) NOT a DSL bug -- local_tile honors the type we pass.
        bidx64 = cutlass.Int64(bidx)
        gX = cute.local_tile(mX, tiler_mn, (bidx64, cluster_y))
        dyn_n = const_expr(self.dyn_n_vec is not None)
        if const_expr(dyn_n):
            # BUCKET mode: mX's extents are dynamic, so the identity tensor (which
            # must be static for the fragment partitions) covers the TILE only.
            # Coordinates are tile-local; the global row is rebuilt from bidx below
            # and the single column tile makes tile-local col == global col.
            cX = cute.make_identity_tensor(tiler_mn)
        else:
            idX = cute.make_identity_tensor(shape)
            cX = cute.local_tile(idX, tiler_mn, (bidx64, cluster_y))

        smem = cutlass.utils.SmemAllocator()
        thr_copy = tiled_copy.get_slice(tidx)
        tXgX = thr_copy.partition_S(gX)
        tXcX = thr_copy.partition_S(cX)
        # Under TMA, mX is a TMA-descriptor tensor, so tXgX is a coordinate partition
        # with no concrete element dtype -> give the register fragment the input dtype
        # explicitly. The direct/cp.async paths keep the dtype-inferring form.
        tXrX = (
            cute.make_rmem_tensor_like(tXgX, dtype=self.dtype)
            if const_expr(self.use_tma)
            else cute.make_rmem_tensor_like(tXgX)
        )

        # Row is fully covered when the cluster_n tiles together span N exactly.
        # Bucket mode is never "even": the tile is the bucket CEILING and the true
        # N is runtime, so the column edge is always dynamically predicated.
        is_even = const_expr(
            False if dyn_n else shape[1] == tiler_mn[1] * self.cluster_n
        )
        if const_expr(dyn_n):
            # Tile-local coordinates: rebuild the global row from the block index.
            # (Single column tile in bucket mode, so tile-local col == global col.)
            row = bidx * tiler_mn[0] + tXcX[(0, None), None, None][0][0]
            # Column predicate vs the RUNTIME N for the tile-edge lanes (the quack
            # rmsnorm ragged idiom, bounds spelled out since the tile is static).
            pv = const_expr(cute.size(tXcX, mode=[0, 1]))
            pk = const_expr(cute.size(tXcX, mode=[2]))
            tXpX = cute.make_rmem_tensor(
                cute.make_layout((pv, 1, pk), stride=(pk, 0, 1)), cutlass.Boolean
            )
            for rest_v in cutlass.range_constexpr(pv):
                for rest_k in cutlass.range_constexpr(pk):
                    tXpX[rest_v, 0, rest_k] = cute.elem_less(
                        tXcX[(0, rest_v), 0, rest_k][1], shape[1]
                    )
        else:
            row = tXcX[(0, None), None, None][0][0]
            tXpX = None
        # Per-thread contiguous copy width in bits. cp.async only supports 32/64/
        # 128-bit transfers, so the wide SMEM-staged path is used only when the
        # vector is >= 32 bits (the perf-critical fp16/bf16 even case: vec*16 =
        # 128). Narrow/ragged (e.g. vec=1 for a prime N) falls back to a direct
        # autovec gmem->rmem, which is correct and memory-safe (compute-sanitizer
        # clean) just not maximally wide -- acceptable off the fast path.
        # Use self.dtype (a concrete numeric class) not mX.element_type: under TMA mX
        # is a descriptor tensor whose element_type is not a plain Numeric.
        cp_bits = const_expr(self.vec * self.dtype.width)
        if const_expr(self.use_tma):
            # TMA-staged load: the TMA unit moves the whole (tiler_m, N) tile
            # gmem->smem in one bulk transfer, driven by the tested PipelineTmaAsync
            # (correct mbarrier count/fence/phase). Then the SAME fragment fold reads
            # sX. The host builder only enables use_tma for the even one-shot case, so
            # is_even holds and OOB rows are handled by the descriptor.
            sX = smem.allocate_tensor(
                self.dtype,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            mbar = smem.allocate_array(cutlass.Int64, num_elems=2)
            tx = const_expr(cute.size(tiler_mn) * self.dtype.width // 8)
            pipe = pipeline.PipelineTmaAsync.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, const_expr(tiled_copy.size)
                ),
                tx_count=tx,
                barrier_storage=mbar,
            )
            pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            tSsX, tSgX = cpasync.tma_partition(
                tma_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(sX, 0, 2),
                cute.group_modes(gX, 0, 2),
            )
            if cute.arch.warp_idx() == 0:
                pipe.producer_acquire(pstate)
                cute.copy(
                    tma_atom, tSgX, tSsX, tma_bar_ptr=pipe.producer_get_barrier(pstate)
                )
                pipe.producer_commit(pstate)
            pipe.consumer_wait(cstate)
            cute.autovec_copy(thr_copy.partition_D(sX), tXrX)
            pipe.consumer_release(cstate)
        elif const_expr((is_even or dyn_n) and cp_bits >= 32 and self.cp_async):
            # SMEM-staged wide async load (the rmsnorm idiom). gmem ptr is marked
            # 128-bit aligned host-side so this emits a true wide global transfer.
            # Bucket mode (dyn_n) stages the same STATIC ceiling tile but the async
            # copy is column-PREDICATED on col < runtime-N (rmsnorm's ragged form);
            # lanes past N stay unloaded and the fold's valid guard discards them.
            # In the even static case tXpX is None -> the plain unpredicated copy.
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            tXsX = thr_copy.partition_D(sX)
            if row < shape[0]:
                copy_utils.copy(tXgX, tXsX, is_async=True, pred=tXpX)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            cute.autovec_copy(tXsX, tXrX)
        elif const_expr(dyn_n):
            # BUCKET mode direct load (narrow vec class): column-PREDICATED copy vs
            # the runtime N (autovec_copy needs static shapes; cute.copy honors
            # pred). Lanes with col >= N stay unloaded -- the fold guard discards.
            if row < shape[0]:
                cute.copy(tiled_copy, tXgX, tXrX, pred=tXpX)
        else:
            # Direct gmem -> rmem (no smem). OOB lanes for ragged N pull data the
            # fold's per-element column guard then discards; memory-safe.
            if row < shape[0]:
                cute.autovec_copy(tXgX, tXrX)

        # Convert the WHOLE fragment to the accumulator dtype in ONE packed op, into
        # a reg fragment with the SAME layout as tXrX. For fp16/bf16 this is the
        # load-bearing perf fix: a per-element scalar conversion caps throughput at
        # ~half bandwidth, while load().to(acc) emits packed conversion over the
        # vector. Keeping the layout lets us index value and coordinate with the SAME
        # (v,rv,m,k) tuple, so argmax indices stay correct.
        acc_dtype = trait.acc  # accumulator dtype (a compile-time Python class)
        tXrXf = cute.make_rmem_tensor(tXrX.layout, acc_dtype)
        tXrXf.store(tXrX.load().to(acc_dtype))

        acc = trait.init()
        nvec, nrv = const_expr(tXrX.shape[0][0]), const_expr(tXrX.shape[0][1])
        nrm = const_expr(tXrX.shape[1])
        nrk = const_expr(tXrX.shape[2])
        # Even N (common case) -> branch-free fold. Ragged N -> a tile column past
        # N WRAPS to (row+1, 0) in the identity coordinate, so guard on BOTH the
        # coordinate's row matching this element's intended row AND col < N. The
        # wrapped element reports row+1 != erow, so it is correctly excluded; the
        # column we pass to the trait (argmax index) is then always the true col.
        # Chunk-local -> global column base for the xcta two-stage split (index
        # traits only). col_base = (sub_row % C) * sub_N; C=1 -> col_base=0 (folded
        # away as a compile-time no-op for the normal, non-split case). Compute in the
        # trait's INDEX dtype: for a huge N the global column reaches N (~4e9), so the
        # (sub_row % C) * sub_N product overflows Int32 -- the trait carries Int64 then
        # (traits._idx_sentinel) and this arithmetic must match, or the index wraps
        # negative before it ever reaches trait.reduce's cast.
        # col_base only matters for INDEX traits (has .idx); non-index traits ignore the
        # index arg entirely, so a bare 0 is fine and avoids calling a None dtype. When
        # it does matter, compute in the trait's index dtype so a huge global column
        # (up to N ~ 4e9) doesn't overflow Int32 before trait.reduce's cast.
        # The `is not None` guards below sit inside const_expr(...), which a type checker
        # cannot see through, so it still reads idx_dtype as Optional and flags the calls
        # as not-callable. The guards are real: a non-index trait takes the `else` branch.
        idx_dtype = getattr(trait, "idx", None)  # compile-time class (Int32/Int64/None)
        if const_expr(self.index_chunks > 1 and idx_dtype is not None):
            # pyrefly: ignore [not-callable]
            col_base = idx_dtype(row % const_expr(self.index_chunks)) * const_expr(
                shape[1]
            )
        elif const_expr(self.index_rebase and idx_dtype is not None):
            # Runtime rebase (bucket mode): C from the c_chunks launch arg, the
            # sub-row length from the input's dynamic column extent.
            # pyrefly: ignore [not-callable]
            col_base = idx_dtype(row % c_chunks) * idx_dtype(shape[1])
        else:
            col_base = 0
        use_base = const_expr(
            self.index_chunks > 1 or (self.index_rebase and idx_dtype is not None)
        )
        for rk in cutlass.range_constexpr(nrk):
            for rm in cutlass.range_constexpr(nrm):
                erow = tXcX[(0, 0), rm, 0][0]
                for rv in cutlass.range_constexpr(nrv):
                    for v in cutlass.range_constexpr(nvec):
                        crd = tXcX[(v, rv), rm, rk]
                        valid = (
                            True
                            if const_expr(is_even)
                            else (crd[0] == erow) and (crd[1] < shape[1])
                        )
                        gcol = crd[1] + col_base if use_base else crd[1]
                        acc = trait.reduce(acc, tXrXf[(v, rv), rm, rk], gcol, valid)

        # Cross-thread reduce along the row: warp shuffle, then smem across warps.
        # Allocate the per-warp reduction buffers from the SAME allocator as sX
        # (after it), so they don't alias the still-live staging tile.
        acc = warp_reduce(trait, acc, threads_per_row)
        warps_per_row = const_expr(threads_per_row // WARP)
        num_warps = const_expr(self._num_threads() // WARP)
        rows_per_block = const_expr(num_warps // warps_per_row)
        if const_expr(self.cluster_n > 1):
            # CLUSTERED: one combined cross-warp + cross-CTA reduce (replaces the
            # block reduce). Init the cluster mbarrier, allocate the (warps_per_row,
            # cluster_n)-moded buffers, cluster_wait so every peer's smem is mapped,
            # then fold. After this every CTA in the cluster holds the full-row
            # result; only rank 0 stores (guarded below).
            cbar = smem.allocate_array(cutlass.Int64, num_elems=1)
            if tidx == 0:
                cute.arch.mbarrier_init(cbar, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cbufs = [
                smem.allocate_tensor(
                    trait.fdtypes[f],
                    cute.make_ordered_layout(
                        (rows_per_block, (warps_per_row, self.cluster_n)),
                        order=(1, 0),
                    ),
                    byte_alignment=8,
                )
                for f in range(trait.nfields)
            ]
            cute.arch.cluster_wait()
            acc = cluster_reduce(
                trait, acc, cbufs, cbar, self.cluster_n, warps_per_row, rows_per_block
            )
        elif const_expr(warps_per_row > 1):
            cute.arch.barrier()  # sX reads done before reusing smem
            bufs = [
                smem.allocate_tensor(
                    trait.fdtypes[f], cute.make_layout(num_warps), byte_alignment=8
                )
                for f in range(trait.nfields)
            ]
            # Reduce within each row's warp group only (multi-row blocks must not
            # mix rows' warps -- that was the rows/blk=2 corruption).
            acc = block_reduce(trait, acc, bufs, warps_per_row, rows_per_block)

        # Lane 0 of each row (column coordinate 0) writes the output. final ->
        # project (true row length project_n) and store one result; not-final
        # (multi-CTA stage 1) -> store RAW per-field accumulators for stage 2.
        # `final` is a compile-time constant so it is the OUTER (constexpr) branch;
        # only the store address is data-dependent (the inner dynamic guard).
        # Trait calls stay OUT of the dynamic if (they leak the trait object into
        # the IR flattener) -- compute values first, store under the guard.
        # Only cluster rank 0 (which owns column tile 0, hence local col 0 == global
        # col 0) stores; the other peers computed the same full-row result via the
        # cluster reduce but must not double-write. cluster_n==1 -> the extra guard
        # is a compile-time True.
        col0 = tXcX[(0, 0), 0, 0][1]
        rank0 = const_expr(True) if const_expr(self.cluster_n == 1) else cluster_y == 0
        store = col0 == 0 and row < shape[0] and rank0
        if const_expr(self.final):
            # Bucket mode: the divisor is the RUNTIME row length (self.project_n
            # is the static ceiling there, wrong for interior N).
            if const_expr(dyn_n):
                projected = trait.project(acc, acc_dtype(shape[1]))
            else:
                projected = trait.project(acc, acc_dtype(const_expr(self.project_n)))
            # nouts==1 -> project returns a scalar; nouts==2 (max.dim/min.dim) ->
            # project returns (value, index), stored to mOuts[0]/mOuts[1]. This is
            # the ONE-SHOT path only: the row is not split, so the index is already
            # the true per-row column (no cross-chunk global-index remap needed).
            if const_expr(self.nouts == 1):
                result = mOuts[0].element_type(projected)
                if store:
                    mOuts[0][row] = result
            else:
                results = [
                    mOuts[f].element_type(projected[f])
                    for f in range(const_expr(self.nouts))
                ]
                if store:
                    for f in cutlass.range_constexpr(self.nouts):
                        mOuts[f][row] = results[f]
        else:
            vals = [trait.fdtypes[f](acc[f]) for f in range(trait.nfields)]
            if store:
                for f in cutlass.range_constexpr(trait.nfields):
                    mOuts[f][row] = vals[f]


_CACHE = {}


def _cute_aligned(t, align_bytes):
    # Mark the gmem pointer as `align_bytes`-aligned so the DSL may emit wide
    # (128-bit) cp.async loads. from_dlpack defaults to the element's natural
    # alignment (2 B for fp16), which forces narrow loads and halves bandwidth.
    # torch allocations are >=256 B aligned, and even-N row stride is a multiple
    # of the vector width, so 16 B is safe here.
    # enable_tvm_ffi: fast torch->tvm-ffi C exchange (~0.8us vs ~3.6us capsule).
    # read-only: only called for the INPUT (via _aligned_in), so a COW input exports
    # without materializing (see launch._ro).
    ct = cute.runtime.from_dlpack(
        _L.ReadOnlyTensorWrapper(t), assumed_align=align_bytes, enable_tvm_ffi=True
    )
    ct.element_type = _L.torch2cute[t.dtype]
    return ct


def _align_bytes(x, op):
    # 16-byte alignment enables the 128-bit load only when the row stride keeps it
    # (even N); for ragged N the second row may be misaligned, so assume just the
    # element width and let the per-element-guarded narrow path handle it.
    N = x.shape[-1]
    vec = math.gcd(N, 128 // (x.element_size() * 8))
    return (
        (vec * x.element_size())
        if (N % (vec * op._threads_per_row()) == 0)
        else x.element_size()
    )


def _aligned_in(x, op):
    return _cute_aligned(x, _align_bytes(x, op))


def _aligned_in_dynM(x, op):
    # DYNAMIC-M input wrapper (mode 0 = rows dynamic, N static). One compiled kernel
    # serves any M at this N -- no recompile per batch size. N stays static so the
    # kernel's const_expr vec/tile checks resolve; alignment is N-derived (static).
    return _L.cute_tensor_dynM(x, align=_align_bytes(x, op), ndim=2, read_only=True)


def _bucket_ceiling(N, elsize):
    # Static tile ceiling for BUCKET mode: smallest HALF-OCTAVE rung (2^k or 3*2^k,
    # floor 256) >= N, or None when the ceiling tile would blow the one-shot smem
    # budget (caller then compiles the exact-N kernel as before). Half-octave rungs
    # cap the tile's wasted-lane fraction at 25% (a pure power-of-two ladder wastes
    # up to 50% -- measured ~20% slower at N just past a power of two) while keeping
    # the bucket count logarithmic; both rung forms divide cleanly by the legal
    # vec*tpr powers of two (3*2^k keeps a large 2-adic factor).
    p = 1 << (N - 1).bit_length()
    half = (3 * p) // 4
    c = max(256, half if half >= N else p)
    return c if c * elsize <= _ROW_SMEM_BUDGET else None


# One-shot smem tile budget (mirrors kernel_general._SMEM_BUDGET, which routes
# larger N to the xcta split before reduce_row is ever called).
_ROW_SMEM_BUDGET = 192 * 1024


def _reduce_row_bucket(trait, trait_key, x, out_dtypes, vec, ceiling, cp_async):
    # BUCKET-mode one-shot row reduce: ONE compiled kernel per (vec-class, ceiling)
    # serves every ragged N in the bucket (dyn-N wrap + column-predicated loads +
    # runtime project divisor -- see RowReduce.dyn_n_vec). Replaces the long tail
    # of per-exact-N ragged compiles; the branch-free even-N kernels keep their
    # exact-N compiles (they are the perf anchors and genuinely differ per N).
    # Serves 1- and 2-output traits (max.dim/min.dim): bucket coordinates are
    # tile-local but the single column tile makes tile-local col == global col,
    # so projected indices are already correct.
    M, N = x.shape
    nouts = len(out_dtypes)
    outs = [torch.empty(M, device=x.device, dtype=d) for d in out_dtypes]
    key = (
        "rowb",
        trait_key,
        x.dtype,
        tuple(out_dtypes),
        vec,
        ceiling,
        trait.nfields,
        cp_async,
    )
    mouts = [_L.cute_tensor_dynM(o, ndim=1) for o in outs]

    def _build():
        op = RowReduce(
            trait,
            _L.torch2cute[x.dtype],
            ceiling,
            nouts=nouts,
            cp_async=cp_async,
            dyn_n_vec=vec,
        )
        # Every row starts at a multiple of N*elsize with N % vec == 0, so vec-wide
        # alignment holds for every row; the predicated copies are vec-wide.
        align = vec * x.element_size()
        xin = _L.cute_tensor_dynMN(x, vec, align=align, read_only=True)
        fn = _compile(op, xin, mouts, _stream())
        return (fn, align)

    fn, align = cached_plan(_CACHE, key, _build, op=f"aten::{trait_key}")
    xin = _L.cute_tensor_dynMN(x, vec, align=align, read_only=True)
    fn(xin, mouts, _stream())
    return outs


def _cap_cluster_n(N, vec, tpr, want, device):
    # Cap the requested cluster_n to what is legal + supported:
    #   - sm_90+ only (Ampere/Ada lack cluster support); Blackwell caps at 8, else 16.
    #   - each peer CTA must own a distinct, non-empty column tile: tpr*cluster_n must
    #     not exceed the row's vector-block count (N//vec) -- mirrors quack _cap_cluster_n.
    if want <= 1:
        return 1
    major = torch.cuda.get_device_properties(device).major
    if major < 9:
        return 1
    hw_max = 8 if major == 12 else 16
    tpr = tpr if tpr is not None else 1
    max_by_tile = max(1, (N // vec) // max(tpr, 1))
    return max(1, min(want, hw_max, max_by_tile))


def reduce_row(
    trait,
    trait_key,
    x,
    out_dtype,
    block=None,
    tpr=None,
    nt=None,
    cp_async=True,
    cluster_n=1,
    use_tma=False,
):
    assert x.dim() == 2 and x.is_cuda and x.stride(-1) == 1  # noqa: S101
    M, N = x.shape
    vec = math.gcd(N, 128 // (x.element_size() * 8))
    # RAGGED N routes to BUCKET mode (one kernel per (vec-class, power-of-two
    # ceiling) serves every N in the bucket): ragged rows were both the recompile
    # long tail (every distinct N compiled its own kernel) and already on the slow
    # guarded path, so the bucket's predication costs nothing extra. EVEN N keeps
    # its exact-N branch-free compile -- those are the perf anchors, and even-N
    # values cluster (powers of two, model dims) so their kernel count stays small.
    # Only on the plain path (no knob overrides -- autotuner keys must stay exact).
    cfg = _choose_config(N, x.element_size() * 8, trait.nfields)
    plain = tpr is None and nt is None and cluster_n == 1 and not use_tma
    if plain and N % (vec * cfg.tpr) != 0:
        ceiling = _bucket_ceiling(N, x.element_size())
        if ceiling is not None:
            # Direct predicated loads, NOT cp.async: on the bucket path the
            # smem-staged async copy loses or ties on every measured shape (B200
            # graph-mode sweep: e.g. 512x6143 4.1 vs 6.2us, 8192x513 4.1 vs 6.2)
            # -- the ceiling tile stages dead columns through smem while the
            # direct path's predication skips them at the source.
            return _reduce_row_bucket(
                trait, trait_key, x, [out_dtype], vec, ceiling, False
            )[0]
    out = torch.empty(M, device=x.device, dtype=out_dtype)
    cn = _cap_cluster_n(N, vec, tpr, cluster_n, x.device)
    # TMA gate: the bulk tensor load moves the whole static (tiler_m, N) tile, so it
    # needs an EVEN tile (N a clean multiple of vec*tpr) and is exclusive with the
    # cluster split. Silently fall back to the cp.async/direct path otherwise so the
    # knob never produces a wrong or illegal launch.
    eff_tpr = tpr if tpr is not None else 0
    even_tile = (eff_tpr == 0) or (N % (vec * eff_tpr) == 0)
    tma = bool(use_tma) and cn == 1 and even_tile
    if tma and torch.cuda.get_device_properties(x.device).major < 9:
        tma = False  # TMA is sm_90+
    # DYNAMIC M: only N (+dtype/trait) keys the kernel; the grid reads mX.shape[0]
    # and the M/output extents are dynamic, so one compile serves all M. The input
    # alignment is also a pure function of (N, dtype), so it is cached next to the
    # compiled fn -- on the hot path we skip both the op construction and the
    # _align_bytes/_threads_per_row math (only the per-call tensor wrap + launch).
    # tpr/nt are the exposed occupancy knobs (threads-per-row / threads-per-block);
    # cp_async gates the smem-staged load path; cluster_n splits the row across a
    # launch cluster. All key the plan. None/True/1 -> the tuned defaults.
    key = (
        "row",
        trait_key,
        x.dtype,
        out_dtype,
        N,
        trait.nfields,
        tpr,
        nt,
        cp_async,
        cn,
        tma,
    )

    def _build():
        op = RowReduce(
            trait,
            _L.torch2cute[x.dtype],
            N,
            tpr_override=tpr,
            nt_override=nt,
            cp_async=cp_async,
            cluster_n=cn,
            use_tma=tma,
        )
        align = _align_bytes(x, op)
        xin = _L.cute_tensor_dynM(x, align=align, ndim=2, read_only=True)
        fn = _compile(op, xin, [_L.cute_tensor_dynM(out, ndim=1)], _stream())
        return (fn, align)

    fn, align = cached_plan(_CACHE, key, _build, op=f"aten::{key[1]}")
    xin = _L.cute_tensor_dynM(x, align=align, ndim=2, read_only=True)
    fn(xin, [_L.cute_tensor_dynM(out, ndim=1)], _stream())
    return out


def reduce_row_2out(trait, trait_key, x, out_dtypes, block=None):
    # Two-output one-shot row reduction (max.dim/min.dim: value + index). Same K1
    # kernel as reduce_row but the `final` store writes both projected outputs. Used
    # ONLY on the one-shot (smem-fits) path: the row is not split, so the projected
    # index is already the true per-row column with no global-index remap.
    assert x.dim() == 2 and x.is_cuda and x.stride(-1) == 1  # noqa: S101
    M, N = x.shape
    # RAGGED N -> bucket mode, same gate as reduce_row (bucket coords are
    # tile-local == global here, so the projected index is still the true column).
    vec = math.gcd(N, 128 // (x.element_size() * 8))
    cfg = _choose_config(N, x.element_size() * 8, trait.nfields)
    if N % (vec * cfg.tpr) != 0:
        ceiling = _bucket_ceiling(N, x.element_size())
        if ceiling is not None:
            # Direct loads (see the reduce_row bucket gate note).
            res = _reduce_row_bucket(
                trait, trait_key, x, list(out_dtypes), vec, ceiling, False
            )
            return tuple(res)
    outs = [torch.empty(M, device=x.device, dtype=d) for d in out_dtypes]
    key = ("row_2out", trait_key, x.dtype, tuple(out_dtypes), N, trait.nfields)
    mouts = [_L.cute_tensor_dynM(o, ndim=1) for o in outs]

    def _build():
        op = RowReduce(trait, _L.torch2cute[x.dtype], N, nouts=2)
        align = _align_bytes(x, op)
        xin = _L.cute_tensor_dynM(x, align=align, ndim=2, read_only=True)
        fn = _compile(op, xin, mouts, _stream())
        return (fn, align)

    fn, align = cached_plan(_CACHE, key, _build, op=f"aten::{key[1]}")
    xin = _L.cute_tensor_dynM(x, align=align, ndim=2, read_only=True)
    fn(xin, mouts, _stream())
    return tuple(outs)
