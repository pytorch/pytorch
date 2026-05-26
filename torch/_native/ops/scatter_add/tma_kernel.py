"""TMA-based scatter_add for the ``index.unsqueeze(-1).expand(-1, N)`` pattern.

Port of the CUDA C++ kernel from https://github.com/pytorch/pytorch/pull/182675
to CuTeDSL. Each CTA holds ``_WARPS_PER_CTA`` warps; each warp independently
handles its own row stream (within an assigned chunk-index range): a tile-mode
TMA bulk load (``cute.copy`` with ``CopyBulkTensorTileG2SOp`` via
``make_tiled_tma_atom``) stages ``src[i, d_start:d_end]`` into the warp's smem
slice, then ``cp.reduce.async.bulk.global.shared::cta.bulk_group.add``
deposits the reduction into ``out[index[i], d_start:d_end]``.

Using the tile-mode TMA with a descriptor over the full ``(M_src, N)``
source tensor lets TMA clamp out-of-range column reads to zero, so the
final partial chunk (when ``N`` isn't a multiple of ``chunk_elems``) is
handled natively -- we just reduce only the valid byte count on the
store side. That widens coverage to any ``row_bytes`` that's 16-byte
aligned (the ``cp.reduce.async.bulk`` gmem operand requirement).

Synchronization between load and reduce goes through
``PerWarpTmaPipeline`` (in ``_per_warp_pipeline.py``), an
API-compatible drop-in for ``cutlass.pipeline.PipelineTmaAsync`` that
supports N independent pipes per CTA -- one per warp. Stock
``PipelineTmaAsync.create`` gates ``mbarrier_init`` on
``warp_idx == 0`` and so cannot service multiple pipes; the wrapper
inits mbarriers from lane 0 of every warp instead.

The pipeline is software-pipelined across the flat sequence of
``(entry, chunk)`` pairs per warp: each loop iteration issues the TMA
load for the current pair and consumes the previous one, keeping one
TMA load in flight alongside an in-progress bulk-reduce. An epilogue
drains the final outstanding load via ``drain_bulk_reduces``.

Small-N path: when the entire row fits in one chunk
(``num_chunks == 1``, equivalent to ``row_bytes <= 512``), the kernel
switches to a multi-row tile pattern: each pipeline stage holds
``_ROWS_PER_STAGE_SMALL_N`` rows loaded by a single 2D TMA descriptor
(``cta_tiler=(K, N)``). This amortizes per-descriptor TMA overhead
across K rows, giving up to ~2x speedup on bf16/fp16 small-N shapes
where individual row payloads would underutilize the TMA engine.

Chunks along D at a compile-time ``chunk_elems``: small rows travel as
a single chunk, large rows chunk at 512 B. The TMA descriptor OOB-clamp
handles rows whose length isn't a multiple of ``chunk_elems``.

Grid layout is 2D to maintain SM utilization for shapes with small N
and large D: ``(grid_x, grid_y)``, where ``grid_y`` partitions the
chunk-index axis. When N is large the host sets ``grid_y = 1`` so the
inner chunk loop recovers the classic schedule. When N is tiny (say
100 rows, D=640K), ``grid_y`` grows so the grid still fills the
machine. Row mapping: warp ``w`` in CTA ``(bx, by)`` starts at row
``bx * _WARPS_PER_CTA + w`` and strides by ``gdim_x * _WARPS_PER_CTA``.

Restrictions (enforced by the host cond in ``cutedsl_impl.py``):
  - sm_90+ (cp.reduce.async.bulk availability)
  - dim == 0, rank >= 2, self/src inner-contiguous
  - index is the expanded-1D pattern (same shape as src, stride 0 on every
    axis except 0)
  - dtype in {fp32, fp16, bf16}
  - ``row_bytes % 16 == 0`` (for both the TMA load operand and the
    cp.reduce.async.bulk gmem operand)
"""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.cute.testing as cute_testing
from cutlass import BFloat16, const_expr, Float16, Float32, Int32, Int64

import torch
from torch._vendor.quack.cache_utils import jit_cache

from ._per_warp_pipeline import PerWarpTmaPipeline
from ._ptx import cvta_smem, make_bulk_reduce_add


_MAX_CHUNK_BYTES = 512
_WARPS_PER_CTA = 8
_THREADS_PER_CTA = _WARPS_PER_CTA * 32
_NUM_STAGES = 2
# cp.async.bulk requires 128B-aligned smem destinations.
_SMEM_ALIGN_BYTES = 128


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


# When num_chunks == 1 (small-N: row_bytes <= 512), each pipeline stage
# holds _ROWS_PER_STAGE_SMALL_N rows loaded with one 2D TMA descriptor
# (cta_tiler=(K, N)) for better TMA utilization on small payloads.
#
# K=4 is chosen empirically. K-sweep on B200 (cache-resident, batched
# event timing) for fp32/bf16 D=128 across uniform/skewed contention:
# K=2 leaves TMA-descriptor overhead on the table; K=8 starts to spill
# smem under _WARPS_PER_CTA=8 * _NUM_STAGES=2 at D >= 256 and regresses
# bf16. K=4 is best across the sweep with no spills at D <= 256
# (per-warp slab size = K * N * _NUM_STAGES * elem_bytes; at K=4,
# _WARPS_PER_CTA=8, _NUM_STAGES=2, D=256, fp32 -> 64KB which fits the
# 100KB-per-CTA smem budget; bf16/fp16 are half that).
_ROWS_PER_STAGE_SMALL_N = 4


_TORCH_TO_CUTE = {
    torch.float32: Float32,
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
}


_bulk_reduce_add_f32 = make_bulk_reduce_add("f32")
_bulk_reduce_add_f16 = make_bulk_reduce_add("noftz.f16")
_bulk_reduce_add_bf16 = make_bulk_reduce_add("noftz.bf16")


def _reduce_op_for(dtype):
    if dtype is Float32:
        return _bulk_reduce_add_f32
    if dtype is Float16:
        return _bulk_reduce_add_f16
    if dtype is BFloat16:
        return _bulk_reduce_add_bf16
    raise ValueError(f"unsupported dtype: {dtype}")


def _make_kernel(
    dtype,
    elem_bytes: int,
    N: int,
    chunk_elems: int,
    reduce_op,
    contig: bool,
):
    """Build a dtype-specialized kernel closure. dtype/elem_bytes/N
    /chunk_elems/contig are Python-time constants that the preprocessor
    folds at cute.compile time.

    ``chunk_elems`` is baked in at compile time. The TMA descriptor
    built in ``_launch`` (via ``make_tiled_tma_atom`` on the source
    tensor) enables OOB-clamp-to-zero for reads past column ``N``; the
    reduce side then writes only the actual valid byte count, so
    partial final chunks are handled natively.

    Each CTA holds ``_WARPS_PER_CTA`` warps; each warp owns its own
    smem buffer slice and mbarrier slice (``_NUM_STAGES`` mbarriers
    per warp, TMA full-side only). Lane 0 of each warp initializes
    its mbarriers and drives both the TMA load (producer) and the
    bulk-reduce (consumer) for its row stream.

    ``contig`` gates a fast path on the reduce side: when True, the
    output row stride is D at compile time, avoiding a runtime stride
    multiply. When False the kernel takes a runtime
    ``out_row_stride`` arg so outer-strided outputs (e.g. slices) work.
    The TMA load always goes through the descriptor and doesn't use
    ``contig``.
    """

    chunk_bytes = chunk_elems * elem_bytes
    # Compile-time derived values. num_chunks covers row_bytes with
    # partial final chunk handled by TMA OOB clamp.
    num_chunks = (N + chunk_elems - 1) // chunk_elems
    # Small-N path: row fits in one TMA chunk (num_chunks == 1). Each
    # pipeline stage holds K rows loaded with a single 2D TMA tile
    # (cta_tiler=(K, N)). Better TMA descriptor utilization than 1
    # row/stage on small payloads, especially bf16/fp16. Falls back to
    # the per-row-chunked path when num_chunks > 1.
    small_n = num_chunks == 1
    K = _ROWS_PER_STAGE_SMALL_N if small_n else 1

    # 2-stage pipeline buffer is laid out column-major; stage i starts
    # at offset i * stage_stride_elems * elem_bytes, so the stride must
    # round chunk_bytes up to a multiple of _SMEM_ALIGN_BYTES. Otherwise
    # stage 1 is misaligned and the kernel faults the first time a CTA
    # writes it. Bites both chunk_bytes < 128 (small D) and chunk_bytes
    # not a multiple of 128 (e.g. fp32 N=36 -> 144 B).
    stage_stride_elems = _round_up(chunk_elems, _SMEM_ALIGN_BYTES // elem_bytes)
    # Same padding for the small-N tile-stage stride: TMA destination is
    # the next stage's K*N tile, so its base must be 128B-aligned. Bites
    # any K*N*elem_bytes that isn't a 128-multiple (e.g. fp32 N=4 -> 64 B,
    # bf16 N=8 -> 64 B, fp32 N=12 -> 192 B).
    tile_stage_stride_elems = _round_up(K * N, _SMEM_ALIGN_BYTES // elem_bytes)

    @cute.kernel
    def _kernel_small_n(
        tma_atom: cute.CopyAtom,
        tma_tensor_src: cute.Tensor,  # 2D TMA-view of mSrc, tile (K, N)
        mIndex: cute.Tensor,
        mOut: cute.Tensor,
        out_row_stride: Int64,
    ):
        # Small-N kernel: K rows per pipeline stage, single 2D TMA load
        # per stage. Smem layout per warp: (K, N, _NUM_STAGES) row-major
        # so each row of the tile is N contiguous elements -- ready for
        # bulk-reduce.
        #
        # Maintenance contract with ``_kernel``: this is a structural
        # specialization (K rows per stage, single 2D TMA descriptor)
        # of the same producer/consumer pipeline. Changes to the shared
        # control flow -- pipeline state machine, mbarrier init, contig
        # vs strided out -- must be made in both. Differences from
        # ``_kernel``:
        #   - smem layout has an extra K dim (row-major (K, N) tile);
        #   - tile_idx counts K-row groups, not single rows;
        #   - the inner reduce loop is unrolled K times with a
        #     ``row < num_entries`` mask for the partial last tile;
        #   - no chunks_per_cta / bidy partitioning (num_chunks == 1).
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim_x, _, _ = cute.arch.grid_dim()

        warp_id = tidx // Int32(32)
        lane_id = tidx % Int32(32)

        num_entries = mIndex.shape[0]

        if const_expr(contig):
            out_rs = Int64(N)
        else:
            out_rs = out_row_stride

        smem = cutlass.utils.SmemAllocator()
        # Per-warp smem: (K, N, _NUM_STAGES) row-major within a stage
        # (stride N, 1), then padded to 128B between stages (and between
        # warps) so each TMA destination starts on a 128B boundary as
        # cp.async.bulk requires.
        sBuf_all = smem.allocate_tensor(
            dtype,
            cute.make_layout(
                (K, N, _NUM_STAGES, _WARPS_PER_CTA),
                stride=(
                    N,
                    1,
                    tile_stage_stride_elems,
                    tile_stage_stride_elems * _NUM_STAGES,
                ),
            ),
            _SMEM_ALIGN_BYTES,
        )
        mbar_per_warp = PerWarpTmaPipeline.barrier_storage_size(_NUM_STAGES)
        mbar_all = smem.allocate_array(
            cutlass.Uint64, num_elems=mbar_per_warp * _WARPS_PER_CTA
        )

        my_buf_off = warp_id * Int32(tile_stage_stride_elems * _NUM_STAGES)
        my_mbar_off = warp_id * Int32(mbar_per_warp)
        my_buf_ptr = sBuf_all.iterator + my_buf_off
        my_mbar_ptr = mbar_all + my_mbar_off

        tile_bytes = K * N * elem_bytes
        pipe = PerWarpTmaPipeline.create(
            num_stages=_NUM_STAGES,
            barrier_storage=my_mbar_ptr,
            tx_count=tile_bytes,
            lane_id=lane_id,
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # Per-warp TMA partition. Smem view: (K, N, _NUM_STAGES)
        # row-major within a stage, padded between stages.
        my_sBuf = cute.make_tensor(
            my_buf_ptr,
            cute.make_layout(
                (K, N, _NUM_STAGES), stride=(N, 1, tile_stage_stride_elems)
            ),
        )
        # gmem (M, N) tiled by (K, N) -> shape (K, N, num_K_tiles, 1).
        tiled_gmem = cute.local_tile(tma_tensor_src, (K, N), (None, None))
        tma_smem, tma_gmem = cpasync.tma_partition(
            tma_atom,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(my_sBuf, 0, 2),
            cute.group_modes(tiled_gmem, 0, 2),
        )

        prod_state = PerWarpTmaPipeline.make_producer_state(_NUM_STAGES)
        cons_state = PerWarpTmaPipeline.make_consumer_state(_NUM_STAGES)
        prev_tile_idx = Int32(0)
        pair_count = Int32(0)

        n_tiles = (num_entries + Int32(K) - Int32(1)) // Int32(K)
        # Per-warp tile stream: warp w in CTA bx starts at tile
        # bx * _WARPS_PER_CTA + w; strides by gdim_x * _WARPS_PER_CTA.
        tile_idx = bidx * Int32(_WARPS_PER_CTA) + warp_id
        warp_stride = gdim_x * Int32(_WARPS_PER_CTA)

        while tile_idx < n_tiles:
            # Producer: lane 0 acquires (mbarrier_arrive_count=1);
            # cute.copy is called by all 32 lanes (atom elects
            # internally per CuTeDSL contract).
            if lane_id == Int32(0):
                pipe.producer_acquire(prod_state)
            cute.copy(
                tma_atom,
                tma_gmem[None, tile_idx, Int32(0)],
                tma_smem[None, prod_state.index],
                tma_bar_ptr=pipe.producer_get_barrier(prod_state),
            )
            pipe.producer_commit(prod_state)
            prod_state.advance()

            if pair_count > Int32(0):
                pipe.consumer_wait(cons_state)
                stage_off = cons_state.index * Int32(tile_stage_stride_elems)

                if lane_id == Int32(0):
                    base_row = prev_tile_idx * Int32(K)
                    for k in cutlass.range_constexpr(K):
                        row = base_row + Int32(k)
                        if row < num_entries:
                            r = Int64(mIndex[row])
                            cute_testing.assert_(r >= Int64(0))
                            cute_testing.assert_(r < Int64(mOut.shape[0]))
                            dst_off = r * out_rs
                            gmem_dst_u64 = Int64((mOut.iterator + dst_off).toint())
                            reduce_op(
                                gmem_dst_u64,
                                cvta_smem(my_buf_ptr + stage_off + Int32(k * N)),
                                Int32(N * elem_bytes),
                            )
                    cute.arch.cp_async_bulk_commit_group()
                    pipe.consumer_release(cons_state)
                cons_state.advance()

            prev_tile_idx = tile_idx
            pair_count = pair_count + Int32(1)
            tile_idx = tile_idx + warp_stride

        # Epilogue: drain final tile.
        if pair_count > Int32(0):
            pipe.consumer_wait(cons_state)
            stage_off = cons_state.index * Int32(tile_stage_stride_elems)

            if lane_id == Int32(0):
                base_row = prev_tile_idx * Int32(K)
                for k in cutlass.range_constexpr(K):
                    row = base_row + Int32(k)
                    if row < num_entries:
                        r = Int64(mIndex[row])
                        cute_testing.assert_(r >= Int64(0))
                        cute_testing.assert_(r < Int64(mOut.shape[0]))
                        dst_off = r * out_rs
                        gmem_dst_u64 = Int64((mOut.iterator + dst_off).toint())
                        reduce_op(
                            gmem_dst_u64,
                            cvta_smem(my_buf_ptr + stage_off + Int32(k * N)),
                            Int32(N * elem_bytes),
                        )
                cute.arch.cp_async_bulk_commit_group()
                pipe.consumer_release(cons_state)
                pipe.drain_bulk_reduces()

    @cute.kernel
    def _kernel(
        tma_atom: cute.CopyAtom,
        tma_tensor_src: cute.Tensor,  # TMA-view of mSrc
        mIndex: cute.Tensor,
        mOut: cute.Tensor,
        chunks_per_cta: Int32,
        out_row_stride: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdim_x, _, _ = cute.arch.grid_dim()

        warp_id = tidx // Int32(32)
        lane_id = tidx % Int32(32)

        # num_entries (M_src) comes from mIndex's shape; mIndex is 1D
        # of length M_src after host-side flattening.
        num_entries = mIndex.shape[0]

        # Chunk-index range assigned to this CTA. bidy partitions the
        # chunk axis into disjoint slices of chunks_per_cta; when
        # chunks_per_cta == num_chunks, grid_y == 1 and every CTA sees
        # the whole D.
        chunk_start = bidy * chunks_per_cta
        chunk_end = chunk_start + chunks_per_cta
        if chunk_end > Int32(num_chunks):
            chunk_end = Int32(num_chunks)

        # Reduce-side out-row stride: compile-time N (contig) or runtime.
        if const_expr(contig):
            out_rs = Int64(N)
        else:
            out_rs = out_row_stride

        smem = cutlass.utils.SmemAllocator()
        # Per-warp slot: _NUM_STAGES tiles of chunk_elems, total
        # _WARPS_PER_CTA * _NUM_STAGES tiles.
        sBuf_all = smem.allocate_tensor(
            dtype,
            cute.make_layout(
                (chunk_elems, _NUM_STAGES * _WARPS_PER_CTA),
                stride=(1, stage_stride_elems),
            ),
            _SMEM_ALIGN_BYTES,
        )
        # Per-warp barrier slice: full + empty mbarriers per stage.
        mbar_per_warp = PerWarpTmaPipeline.barrier_storage_size(_NUM_STAGES)
        mbar_all = smem.allocate_array(
            cutlass.Uint64, num_elems=mbar_per_warp * _WARPS_PER_CTA
        )

        # Padded stage stride keeps every stage 128B-aligned for cp.async.bulk;
        # propagate it into the per-warp offset / view so warp w lands on its
        # own 2-tile slice rather than overlapping the next warp's.
        my_buf_off = warp_id * Int32(_NUM_STAGES * stage_stride_elems)
        my_mbar_off = warp_id * Int32(mbar_per_warp)
        my_buf_ptr = sBuf_all.iterator + my_buf_off
        my_mbar_ptr = mbar_all + my_mbar_off

        # One independent pipe per warp. PipelineTmaAsync.create's
        # mbarrier_init is gated on warp 0 only, which can't service N
        # pipes per CTA -- PerWarpTmaPipeline replaces it with a
        # lane-0-of-each-warp init.
        pipe = PerWarpTmaPipeline.create(
            num_stages=_NUM_STAGES,
            barrier_storage=my_mbar_ptr,
            tx_count=chunk_bytes,
            lane_id=lane_id,
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # Per-warp TMA partition. Each warp views only its own smem
        # slice as a (chunk_elems, _NUM_STAGES) tensor.
        my_sBuf = cute.make_tensor(
            my_buf_ptr,
            cute.make_layout(
                (chunk_elems, _NUM_STAGES), stride=(1, stage_stride_elems)
            ),
        )
        tiled_gmem = cute.local_tile(tma_tensor_src, (1, chunk_elems), (None, None))
        tma_smem, tma_gmem = cpasync.tma_partition(
            tma_atom,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(my_sBuf, 0, 1),
            cute.group_modes(tiled_gmem, 0, 2),
        )

        # Software-pipelined schedule (per-warp): each iteration issues
        # the TMA load for the current (entry, chunk) pair, then
        # consumes the previous pair (bulk-reduce). With _NUM_STAGES=2
        # this keeps one TMA load in flight while a bulk-reduce runs.
        # Final iteration's load is drained in an epilogue.
        prod_state = PerWarpTmaPipeline.make_producer_state(_NUM_STAGES)
        cons_state = PerWarpTmaPipeline.make_consumer_state(_NUM_STAGES)
        prev_chunk_idx = Int32(0)
        prev_r = Int64(0)
        # pair_count is a runtime Int32 (not a Python bool) so the
        # ``pair_count > 0`` branches stay in the compiled IR.
        pair_count = Int32(0)

        # Per-warp row stream: warp w in CTA bx starts at
        # bx * _WARPS_PER_CTA + w, strides by gdim_x * _WARPS_PER_CTA.
        base = bidx * Int32(_WARPS_PER_CTA) + warp_id
        warp_stride = gdim_x * Int32(_WARPS_PER_CTA)
        while base < num_entries:
            entry_id = base

            chunk_idx = chunk_start
            while chunk_idx < chunk_end:
                # All 32 lanes execute the loop body in lockstep. The
                # mbarrier ops with arrive_count=1 (producer_acquire,
                # consumer_release) are gated on lane 0 so we don't
                # over-arrive. Per-thread ops (consumer_wait,
                # mbarrier_wait inside) and the cute.copy TMA load
                # (which elects internally per CuTeDSL contract) run
                # on all 32 lanes -- the warp must stay converged at
                # the elect.sync inside the TMA atom, otherwise
                # divergent lanes deadlock waiting for elected lane.
                # Driver-thread-only ops (bulk-reduce gmem issue) stay
                # lane-0-gated. mIndex[entry_id] is read by all lanes
                # (uniform load, coalesces) but the bounds assert is
                # lane-0-only -- 32 redundant trap checks per row would
                # be 32x the assertion cost under --enable-assertions.
                r = Int64(mIndex[entry_id])
                if lane_id == Int32(0):
                    cute_testing.assert_(r >= Int64(0))
                    cute_testing.assert_(r < Int64(mOut.shape[0]))

                # Lane-0-only producer_acquire: lane 0 spins on the
                # empty mbarrier while lanes 1-31 race ahead. Warp
                # convergence is restored at the elect.sync inside
                # cute.copy's TMA atom (CuTeDSL contract: TMA copy elects
                # one issuing lane internally). PerWarpTmaPipeline's
                # bare mbarrier_arrive_and_expect_tx is per-thread-safe
                # in this divergent context (stock PipelineTmaAsync's
                # elect_one wrapper would deadlock here).
                if lane_id == Int32(0):
                    pipe.producer_acquire(prod_state)
                cute.copy(
                    tma_atom,
                    tma_gmem[None, entry_id, chunk_idx],
                    tma_smem[None, prod_state.index],
                    tma_bar_ptr=pipe.producer_get_barrier(prod_state),
                )
                pipe.producer_commit(prod_state)  # no-op for TMA
                prod_state.advance()

                if pair_count > Int32(0):
                    pipe.consumer_wait(cons_state)
                    cbuf_ptr = my_sBuf[None, cons_state.index].iterator
                    if lane_id == Int32(0):
                        off = prev_chunk_idx * Int32(chunk_elems)
                        cur_elems = Int32(N) - off
                        if cur_elems > Int32(chunk_elems):
                            cur_elems = Int32(chunk_elems)
                        cur_bytes = cur_elems * Int32(elem_bytes)

                        dst_off = prev_r * out_rs + Int64(off)
                        gmem_dst_u64 = Int64((mOut.iterator + dst_off).toint())
                        reduce_op(gmem_dst_u64, cvta_smem(cbuf_ptr), cur_bytes)
                        # commit_group makes smem reuse safe: cp.reduce.
                        # async.bulk reads smem at issue time, and the
                        # bulk-group counter tracks gmem-write completion
                        # separately. The empty-mbarrier arrive on
                        # consumer_release is therefore safe immediately
                        # after commit_group; no per-iteration wait_group
                        # is needed (drain is deferred to kernel exit).
                        cute.arch.cp_async_bulk_commit_group()
                        pipe.consumer_release(cons_state)
                    cons_state.advance()

                prev_chunk_idx = chunk_idx
                prev_r = r
                pair_count = pair_count + Int32(1)

                chunk_idx = chunk_idx + Int32(1)

            base = base + warp_stride

        # Epilogue: drain the last outstanding TMA load + bulk-reduce.
        if pair_count > Int32(0):
            pipe.consumer_wait(cons_state)
            cbuf_ptr = my_sBuf[None, cons_state.index].iterator
            if lane_id == Int32(0):
                off = prev_chunk_idx * Int32(chunk_elems)
                cur_elems = Int32(N) - off
                if cur_elems > Int32(chunk_elems):
                    cur_elems = Int32(chunk_elems)
                cur_bytes = cur_elems * Int32(elem_bytes)

                dst_off = prev_r * out_rs + Int64(off)
                gmem_dst_u64 = Int64((mOut.iterator + dst_off).toint())
                reduce_op(gmem_dst_u64, cvta_smem(cbuf_ptr), cur_bytes)
                cute.arch.cp_async_bulk_commit_group()
                pipe.consumer_release(cons_state)
                pipe.drain_bulk_reduces()

    @cute.jit
    def _launch(
        mSrc: cute.Tensor,
        mIndex: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
        chunks_per_cta: Int32,
        grid_x: Int32,
        grid_y: Int32,
        out_row_stride: Int64,
    ):
        if const_expr(small_n):
            # Small-N: 2D TMA descriptor with cta_tiler=(K, N), one
            # tile per pipeline stage holds K rows. grid_y is always 1
            # in this path (full row in one chunk).
            tma_atom, tma_tensor_src = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mSrc,
                cute.make_layout((K, N), stride=(N, 1)),
                (K, N),
            )
            _kernel_small_n(
                tma_atom,
                tma_tensor_src,
                mIndex,
                mOut,
                out_row_stride,
            ).launch(
                grid=[grid_x, 1, 1],
                block=[_THREADS_PER_CTA, 1, 1],
                stream=stream,
            )
        else:
            # Large-N: 1D TMA descriptor with cta_tiler=(1, chunk_elems).
            # Shape (M_src, N) comes from mSrc; TMA clamps OOB column
            # reads to 0 so rows with N % chunk_elems != 0 are handled
            # natively on the load.
            tma_atom, tma_tensor_src = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mSrc,
                cute.make_layout((1, chunk_elems)),
                (1, chunk_elems),
            )
            _kernel(
                tma_atom,
                tma_tensor_src,
                mIndex,
                mOut,
                chunks_per_cta,
                out_row_stride,
            ).launch(
                grid=[grid_x, grid_y, 1],
                block=[_THREADS_PER_CTA, 1, 1],
                stream=stream,
            )

    return _launch


def _chunk_elems_for(torch_dtype: torch.dtype, N: int) -> int:
    """Compile-time ``chunk_elems``: whole row if it fits in
    ``_MAX_CHUNK_BYTES``, else ``_MAX_CHUNK_BYTES // elem_bytes``.
    Rounded down to a multiple of ``16 / elem_bytes`` so the final
    partial chunk (if any) still satisfies the 16-byte reduce
    alignment."""
    elem_bytes = torch_dtype.itemsize
    row_bytes = N * elem_bytes
    chunk_bytes = min(row_bytes, _MAX_CHUNK_BYTES)
    # Ensure chunk_bytes itself is 16-aligned (true for 512, true for
    # any small row since cond requires row_bytes % 16 == 0).
    return chunk_bytes // elem_bytes


@jit_cache
def _compile_tma_scatter(torch_dtype: torch.dtype, N: int, contig: bool):
    dtype = _TORCH_TO_CUTE[torch_dtype]
    elem_bytes = dtype.width // 8
    chunk_elems = _chunk_elems_for(torch_dtype, N)
    reduce_op = _reduce_op_for(dtype)
    launcher = _make_kernel(dtype, elem_bytes, N, chunk_elems, reduce_op, contig)

    mSrc_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), N), stride=(cute.sym_int64(), 1)
    )
    # Index is guaranteed contiguous by _flatten_for_expanded_1d; fix
    # stride=1 so `mIndex[i]` doesn't emit a runtime stride multiply.
    mIndex_fake = cute.runtime.make_fake_tensor(Int64, (cute.sym_int(),), stride=(1,))
    mOut_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), N), stride=(cute.sym_int64(), 1)
    )
    return cute.compile(
        launcher,
        mSrc_fake,
        mIndex_fake,
        mOut_fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(0),  # chunks_per_cta
        Int32(0),  # grid_x
        Int32(0),  # grid_y
        Int64(0),  # out_row_stride
        # ``--enable-assertions`` keeps the ``cute_testing.assert_``
        # bounds checks on ``r`` live in production. Cost is roughly
        # +1-10% (geomean +7.7%) on most shapes; the safety net is
        # worth it because an OOB ``r`` would otherwise silently
        # corrupt unrelated gmem via ``cp.reduce.async.bulk``.
        options="--enable-tvm-ffi --enable-assertions",
    )


def min_d_divisor_for(dtype: torch.dtype) -> int:
    """Smallest value D must be divisible by so ``cp.reduce.async.bulk``
    gmem operands stay 16-byte aligned: ``D * sizeof(dtype) % 16 == 0``,
    i.e. ``D % (16 / sizeof(dtype)) == 0``. fp32: %4, bf16/fp16: %8.
    Supported dtypes' itemsize divides 16, so plain // suffices.
    """
    return 16 // dtype.itemsize


def row_shape_supported(dtype: torch.dtype, N: int) -> bool:
    """Host-side check: can the TMA kernel handle an N-element row?

    Only requirement is that ``row_bytes = N * elem_size`` is a multiple
    of 16, which is the PTX operand alignment for both the tile-mode
    TMA load (per-row stride must be 16-aligned) and
    ``cp.reduce.async.bulk`` (gmem address + byte count must be
    16-aligned). Rows that aren't multiples of ``chunk_elems`` are
    handled via the TMA descriptor's OOB-clamp-to-zero behavior.
    """
    return (N * dtype.itemsize) % 16 == 0


def _plan_grid(M: int, D: int, chunk_elems: int, sm: int) -> tuple[int, int, int]:
    """Pick ``(grid_x, grid_y, chunks_per_cta)``.

    Strategy: keep the classic 1D schedule (grid_y=1, whole chunk range
    per CTA with internal double-buffering) whenever the row-axis alone
    saturates the GPU. When M is too small for that, split the
    chunk-axis across grid_y so every SM gets work.
    """
    n_chunks = (D + chunk_elems - 1) // chunk_elems
    # Rows per CTA depends on which kernel path runs:
    #   - small_n (num_chunks == 1): each warp's stage holds K rows,
    #     so each CTA's first slice covers _WARPS_PER_CTA * K rows.
    #   - large_n (num_chunks > 1): _WARPS_PER_CTA rows per CTA slice.
    rows_per_cta = _WARPS_PER_CTA * (_ROWS_PER_STAGE_SMALL_N if n_chunks == 1 else 1)
    row_ctas = (M + rows_per_cta - 1) // rows_per_cta
    # Target: keep per-SM CTA count high enough to hide TMA / bulk-reduce
    # latency. sm*32 / sm*64 (matches the prior single-warp heuristic
    # scaled by warps_per_cta = 8) is the geomean optimum on B200.
    target_ctas = sm * 32
    if row_ctas >= target_ctas:
        grid_x = min(row_ctas, sm * 64)
        return grid_x, 1, n_chunks
    # Split the chunk axis until we hit the target. Small-N has
    # n_chunks==1, so the chunk axis can't be split; we just clamp
    # grid_y=1 and accept fewer CTAs.
    if n_chunks == 1:
        return max(1, row_ctas), 1, 1
    want_y = max(1, target_ctas // max(row_ctas, 1))
    grid_y = min(n_chunks, want_y)
    chunks_per_cta = (n_chunks + grid_y - 1) // grid_y
    # Recompute grid_y now that each y-slot holds chunks_per_cta chunks.
    grid_y = (n_chunks + chunks_per_cta - 1) // chunks_per_cta
    grid_x = row_ctas
    return grid_x, grid_y, chunks_per_cta


def tma_scatter_add_into(
    out: torch.Tensor,
    index_1d: torch.Tensor,
    src: torch.Tensor,
) -> None:
    """In-place: ``out[index_1d[i], :] += src[i, :]`` for every i.

    ``out`` / ``src`` are 2D with inner-dim stride 1 (outer row stride
    can differ from N, e.g. a slice of a wider buffer). ``index_1d`` is
    1D int64 of length M_src. ``row_bytes = N * elem_size`` must be a
    multiple of 16; the host cond enforces this.
    """
    M, N = src.shape
    chunk_elems = _chunk_elems_for(src.dtype, N)
    contig = src.stride(0) == N and out.stride(0) == N
    compiled = _compile_tma_scatter(src.dtype, N, contig)
    sm = torch.cuda.get_device_properties(out.device).multi_processor_count

    grid_x, grid_y, chunks_per_cta = _plan_grid(M, N, chunk_elems, sm)
    compiled(
        src,
        index_1d,
        out,
        chunks_per_cta,
        grid_x,
        grid_y,
        out.stride(0),
    )
