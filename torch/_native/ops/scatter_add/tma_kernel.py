"""TMA-based scatter_add for the ``index.unsqueeze(-1).expand(-1, N)`` pattern.

Port of the CUDA C++ kernel from https://github.com/pytorch/pytorch/pull/182675
to CuTeDSL. Each CTA (one warp) handles one source row (within an assigned
chunk-index range): a tile-mode TMA bulk load (``cute.copy`` with
``CopyBulkTensorTileG2SOp`` via ``make_tiled_tma_atom``) stages
``src[i, d_start:d_end]`` into smem, then
``cp.reduce.async.bulk.global.shared::cta.bulk_group.add`` deposits the
reduction into ``out[index[i], d_start:d_end]``.

Using the tile-mode TMA with a descriptor over the full ``(M_src, N)``
source tensor lets TMA clamp out-of-range column reads to zero, so the
final partial chunk (when ``N`` isn't a multiple of ``chunk_elems``) is
handled natively -- we just reduce only the valid byte count on the
store side. That widens coverage to any ``row_bytes`` that's 16-byte
aligned (the ``cp.reduce.async.bulk`` gmem operand requirement).

Synchronization between the load and the reduce uses
``cutlass.pipeline.PipelineTmaAsync`` with ``num_stages=2``: producer
``acquire``/``commit`` guards the TMA issue, consumer ``wait``/``release``
guards the reduce. Producer and consumer cooperative groups are both
``Agent.Thread, size=1`` -- the single driver thread inside the warp
does both roles. ``PipelineState.advance()`` handles the stage index +
phase bit.

The pipeline is software-pipelined across the flat sequence of
``(entry, chunk)`` pairs: each loop iteration issues the TMA load for
the current pair and consumes the previous one, keeping one TMA load
in flight alongside an in-progress bulk-reduce. An epilogue drains
the final outstanding load.

Chunks along D at a compile-time ``chunk_elems``: small rows travel as
a single chunk, large rows chunk at 512 B. The TMA descriptor OOB-clamp
handles rows whose length isn't a multiple of ``chunk_elems``.

Grid layout is 2D to maintain SM utilization for shapes with small N
and large D: ``(grid_x, grid_y)``, where ``grid_y`` partitions the
chunk-index axis. When N is large the host sets ``grid_y = 1`` so the
inner chunk loop recovers the classic schedule. When N is tiny (say
100 rows, D=640K), ``grid_y`` grows so the grid still fills the
machine.

Restrictions (enforced by the host cond in ``cutedsl_impl.py``):
  - sm_90+ (cp.reduce.async.bulk availability)
  - dim == 0, rank >= 2, self/src inner-contiguous
  - index is the expanded-1D pattern (same shape as src, stride 0 on every
    axis except 0)
  - dtype in {fp32, fp16, bf16}
  - ``row_bytes % 16 == 0`` (for both the TMA load operand and the
    cp.reduce.async.bulk gmem operand)
"""

import math

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.cute.testing as cute_testing
import cutlass.pipeline as pipeline
from cutlass import BFloat16, const_expr, Float16, Float32, Int32, Int64

import torch
from torch._vendor.quack.cache_utils import jit_cache

from ._ptx import cvta_smem, make_bulk_reduce_add


_MAX_CHUNK_BYTES = 512
_THREADS_PER_CTA = 32  # one warp per CTA


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
    dtype, elem_bytes: int, N: int, chunk_elems: int, reduce_op, contig: bool
):
    """Build a dtype-specialized kernel closure. dtype/elem_bytes/N
    /chunk_elems/contig are Python-time constants that the preprocessor
    folds at cute.compile time.

    ``chunk_elems`` is baked in at compile time. The TMA descriptor
    built in ``_launch`` (via ``make_tiled_tma_atom`` on the source
    tensor) enables OOB-clamp-to-zero for reads past column ``N``; the
    reduce side then writes only the actual valid byte count, so
    partial final chunks are handled natively.

    One CTA = one warp. The single driver thread (tidx == 0) serves as
    both producer (issues the TMA load) and consumer (issues the
    bulk-reduce). Both CooperativeGroups are ``Agent.Thread, size=1``
    so the mbarrier arrive counts match the single-threaded flow.

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
        sBuf = smem.allocate_tensor(
            dtype,
            cute.make_layout((chunk_elems, 2)),
            128,
        )
        mbar_storage = smem.allocate_array(cutlass.Uint64, num_elems=2 * 2)

        pipe = pipeline.PipelineTmaAsync.create(
            num_stages=2,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1),
            tx_count=chunk_bytes,
            barrier_storage=mbar_storage,
            tidx=tidx,
        )

        # Tile the source (M, N) by (1, chunk_elems) to get
        # (1, chunk_elems, M, num_chunks); tma_partition collapses and
        # returns tma_gmem[None, row_idx, chunk_idx] and tma_smem[None,
        # stage].
        tiled_gmem = cute.local_tile(tma_tensor_src, (1, chunk_elems), (None, None))
        tma_smem, tma_gmem = cpasync.tma_partition(
            tma_atom,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sBuf, 0, 1),
            cute.group_modes(tiled_gmem, 0, 2),
        )

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 2
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 2
        )

        # Software-pipelined schedule: at each iteration, issue the TMA
        # load for the current (entry, chunk) pair, then consume the
        # previous pair (bulk-reduce). With num_stages=2 this keeps one
        # TMA load in flight while a bulk-reduce is running. Final
        # iteration's load is drained in an epilogue after the main loop.
        # pair_count is a runtime Int32 so the ``pair_count > 0`` branch
        # stays in the compiled IR (a Python bool would be baked in at
        # trace time).
        pair_count = Int32(0)
        prev_chunk_idx = Int32(0)
        prev_r = Int64(0)

        base = bidx
        while base < num_entries:
            entry_id = base

            chunk_idx = chunk_start
            while chunk_idx < chunk_end:
                if tidx == Int32(0):
                    r = Int64(mIndex[entry_id])
                    # Bounds check: index values must be valid output
                    # rows. Compiling with ``--enable-assertions`` turns
                    # this into a device-side trap; otherwise it folds
                    # away. Driver thread only -- no need to replicate
                    # the check across all 32 lanes.
                    cute_testing.assert_(r >= Int64(0))
                    cute_testing.assert_(r < Int64(mOut.shape[0]))

                    pipe.producer_acquire(producer_state)
                    cute.copy(
                        tma_atom,
                        tma_gmem[None, entry_id, chunk_idx],
                        tma_smem[None, producer_state.index],
                        tma_bar_ptr=pipe.producer_get_barrier(producer_state),
                    )
                    pipe.producer_commit(producer_state)
                    producer_state.advance()

                    if pair_count > Int32(0):
                        pipe.consumer_wait(consumer_state)
                        cbuf_ptr = sBuf.iterator + consumer_state.index * Int32(
                            chunk_elems
                        )

                        # Partial-chunk handling: actual valid element
                        # count is min(chunk_elems, N - off). TMA
                        # OOB-clamped the tail to 0 in smem, we reduce
                        # only the valid bytes.
                        off = prev_chunk_idx * Int32(chunk_elems)
                        cur_elems = Int32(N) - off
                        if cur_elems > Int32(chunk_elems):
                            cur_elems = Int32(chunk_elems)
                        cur_bytes = cur_elems * Int32(elem_bytes)

                        dst_off = prev_r * out_rs + Int64(off)
                        gmem_dst_u64 = Int64((mOut.iterator + dst_off).toint())
                        reduce_op(gmem_dst_u64, cvta_smem(cbuf_ptr), cur_bytes)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=False)
                        pipe.consumer_release(consumer_state)
                        consumer_state.advance()

                    prev_chunk_idx = chunk_idx
                    prev_r = r
                    pair_count = pair_count + Int32(1)

                chunk_idx = chunk_idx + Int32(1)

            base = base + gdim_x

        # Epilogue: drain the last outstanding TMA load.
        if tidx == Int32(0):
            if pair_count > Int32(0):
                pipe.consumer_wait(consumer_state)
                cbuf_ptr = sBuf.iterator + consumer_state.index * Int32(chunk_elems)

                off = prev_chunk_idx * Int32(chunk_elems)
                cur_elems = Int32(N) - off
                if cur_elems > Int32(chunk_elems):
                    cur_elems = Int32(chunk_elems)
                cur_bytes = cur_elems * Int32(elem_bytes)

                dst_off = prev_r * out_rs + Int64(off)
                gmem_dst_u64 = Int64((mOut.iterator + dst_off).toint())
                reduce_op(gmem_dst_u64, cvta_smem(cbuf_ptr), cur_bytes)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=False)
                pipe.consumer_release(consumer_state)

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
        # Build the tile-mode TMA descriptor. Shape (M_src, N) comes
        # from mSrc; TMA clamps OOB column reads to 0, so rows with
        # ``N % chunk_elems != 0`` are handled natively on the load.
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
    elem_bytes = torch.tensor([], dtype=torch_dtype).element_size()
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
    i.e. ``D % (16 / gcd(16, sizeof(dtype))) == 0``. fp32: %4,
    bf16/fp16: %8.
    """
    esize = torch.tensor([], dtype=dtype).element_size()
    return 16 // math.gcd(16, esize)


def row_shape_supported(dtype: torch.dtype, N: int) -> bool:
    """Host-side check: can the TMA kernel handle an N-element row?

    Only requirement is that ``row_bytes = N * elem_size`` is a multiple
    of 16, which is the PTX operand alignment for both the tile-mode
    TMA load (per-row stride must be 16-aligned) and
    ``cp.reduce.async.bulk`` (gmem address + byte count must be
    16-aligned). Rows that aren't multiples of ``chunk_elems`` are
    handled via the TMA descriptor's OOB-clamp-to-zero behavior.
    """
    esize = torch.tensor([], dtype=dtype).element_size()
    return (N * esize) % 16 == 0


def _plan_grid(M: int, D: int, chunk_elems: int, sm: int) -> tuple[int, int, int]:
    """Pick ``(grid_x, grid_y, chunks_per_cta)``.

    Strategy: keep the classic 1D schedule (grid_y=1, whole chunk range
    per CTA with internal double-buffering) whenever the row-axis alone
    saturates the GPU. When M is too small for that, split the
    chunk-axis across grid_y so every SM gets work.
    """
    n_chunks = (D + chunk_elems - 1) // chunk_elems
    # 1 warp per CTA: need many more CTAs than an 8-warp layout to keep
    # occupancy up. sm*32 target with a sm*64 clamp works well across
    # uniform / high_cont / few_idx on B200.
    row_ctas = M
    target_ctas = sm * 32
    if row_ctas >= target_ctas:
        grid_x = min(row_ctas, sm * 64)
        return grid_x, 1, n_chunks
    # Split the chunk axis until we hit the target.
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
