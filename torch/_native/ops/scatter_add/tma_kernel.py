"""TMA-based scatter_add for the ``index.unsqueeze(-1).expand(-1, N)`` pattern.

Port of the CUDA C++ kernel from https://github.com/pytorch/pytorch/pull/182675
to CuTeDSL. Each warp handles one source row (within an assigned D-range):
a TMA bulk load (``cute.copy`` with ``CopyBulkG2SOp``) stages
``src[i, d_start:d_end]`` into smem, then
``cp.reduce.async.bulk.global.shared::cta.bulk_group.add`` deposits the
reduction into ``out[index[i], d_start:d_end]``.

Chunks along D in ``chunk_bytes``-sized slices, double-buffered so the
next chunk's load overlaps the current chunk's reduce. ``chunk_bytes`` is
``min(row_bytes, _MAX_CHUNK_BYTES)`` at compile time: small rows travel
as a single chunk, large rows chunk at 512B. Because ``cute.copy``
derives the transfer size from the tensor shape, ``chunk_bytes`` must
evenly divide ``row_bytes`` -- the host cond enforces this.

Grid layout is 2D to maintain SM utilization for shapes with small N and
large D: ``(grid_x, grid_y)``, where ``grid_y`` partitions the D axis
into disjoint chunks and each warp iterates only its assigned chunk.
When N is large the host sets ``grid_y = 1`` so the inner chunk loop
recovers the classic schedule. When N is tiny (say 100 rows, D=640K),
``grid_y`` grows so the grid still fills the machine.

Restrictions (enforced by the host cond in ``cutedsl_impl.py``):
  - sm_90+ (cp.reduce.async.bulk availability)
  - dim == 0, rank >= 2, self/src contiguous
  - index is the expanded-1D pattern (same shape as src, stride 0 on every
    axis except 0)
  - dtype in {fp32, fp16, bf16}
  - row_bytes is a multiple of ``min(row_bytes, 512)``
"""

import math

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
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


def _make_kernel(dtype, elem_bytes: int, chunk_elems: int, reduce_op, contig: bool):
    """Build a dtype-specialized kernel closure. dtype/elem_bytes/chunk_elems
    are Python-time constants that the preprocessor folds at cute.compile
    time.

    ``chunk_elems`` is baked in at compile time, so the bulk-load transfer
    size is fixed. That lets us use ``cute.copy(CopyBulkG2SOp(), ...)``
    for the load and ``PipelineTmaAsync`` (tx_count = chunk_bytes) for
    the producer/consumer synchronization. The bulk-reduce side still
    emits raw PTX because cutedsl only ships a tile-mode TMA reduce op,
    which requires a TMA descriptor and doesn't fit our dynamic per-row
    gather.

    One CTA = one warp. The single driver thread (tidx == 0) serves as
    both producer (issues the TMA load) and consumer (issues the
    bulk-reduce). Both CooperativeGroups are ``Agent.Thread, size=1``
    so the mbarrier arrive counts match the single-threaded flow.

    ``contig`` toggles a fast path: when True, row offset is computed as
    ``entry_id * D`` using the compile-time D, avoiding a runtime
    stride multiply. When False the kernel takes runtime row-stride
    arguments so outer-strided tensors (e.g. slices) work.
    """

    chunk_bytes = chunk_elems * elem_bytes

    @cute.kernel
    def _kernel(
        mSrc: cute.Tensor,
        mIndex: cute.Tensor,
        mOut: cute.Tensor,
        num_entries: Int32,
        D: Int32,
        d_tile_size: Int32,
        src_row_stride: Int64,
        out_row_stride: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdim_x, _, _ = cute.arch.grid_dim()

        load_atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), dtype)

        d_start = bidy * d_tile_size
        d_end = d_start + d_tile_size
        if d_end > D:
            d_end = D

        # For the contiguous fast path, row stride is just D; the
        # runtime stride args get ignored.
        if const_expr(contig):
            src_row_stride = Int64(D)
            out_row_stride = Int64(D)

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

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 2
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 2
        )

        base = bidx
        while base < num_entries:
            entry_id = base
            r = Int64(mIndex[entry_id])

            off = d_start
            while off < d_end:
                if tidx == Int32(0):
                    pipe.producer_acquire(producer_state)
                    src_slice = cute.make_tensor(
                        mSrc.iterator + (Int64(entry_id) * src_row_stride + Int64(off)),
                        cute.make_layout(chunk_elems),
                    )
                    dst_slice = cute.make_tensor(
                        sBuf.iterator + producer_state.index * Int32(chunk_elems),
                        cute.make_layout(chunk_elems),
                    )
                    cute.copy(
                        load_atom,
                        src_slice,
                        dst_slice,
                        mbar_ptr=pipe.producer_get_barrier(producer_state),
                    )
                    pipe.producer_commit(producer_state)

                    pipe.consumer_wait(consumer_state)
                    cbuf_ptr = sBuf.iterator + consumer_state.index * Int32(chunk_elems)
                    dst_off = r * out_row_stride + Int64(off)
                    gmem_dst_u64 = Int64((mOut.iterator + dst_off).toint())
                    reduce_op(gmem_dst_u64, cvta_smem(cbuf_ptr), Int32(chunk_bytes))
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=False)
                    pipe.consumer_release(consumer_state)

                producer_state.advance()
                consumer_state.advance()
                off = off + Int32(chunk_elems)

            base = base + gdim_x

    @cute.jit
    def _launch(
        mSrc: cute.Tensor,
        mIndex: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
        num_entries: Int32,
        D: Int32,
        d_tile_size: Int32,
        grid_x: Int32,
        grid_y: Int32,
        src_row_stride: Int64,
        out_row_stride: Int64,
    ):
        _kernel(
            mSrc,
            mIndex,
            mOut,
            num_entries,
            D,
            d_tile_size,
            src_row_stride,
            out_row_stride,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    return _launch


def _chunk_elems_for(torch_dtype: torch.dtype, N: int) -> int:
    """Compile-time ``chunk_elems``: whole row if it fits in
    ``_MAX_CHUNK_BYTES``, else ``_MAX_CHUNK_BYTES // elem_bytes``."""
    elem_bytes = torch.tensor([], dtype=torch_dtype).element_size()
    row_bytes = N * elem_bytes
    chunk_bytes = min(row_bytes, _MAX_CHUNK_BYTES)
    return chunk_bytes // elem_bytes


@jit_cache
def _compile_tma_scatter(torch_dtype: torch.dtype, N: int, contig: bool):
    dtype = _TORCH_TO_CUTE[torch_dtype]
    elem_bytes = dtype.width // 8
    chunk_elems = _chunk_elems_for(torch_dtype, N)
    reduce_op = _reduce_op_for(dtype)
    launcher = _make_kernel(dtype, elem_bytes, chunk_elems, reduce_op, contig)

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
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int64(0),
        Int64(0),
        options="--enable-tvm-ffi",
    )


def min_d_divisor_for(dtype: torch.dtype) -> int:
    """Smallest value D must be divisible by so ``cp.async.bulk`` transfers
    stay 16-byte-aligned: ``D * sizeof(dtype) % 16 == 0``, i.e.
    ``D % (16 / gcd(16, sizeof(dtype))) == 0``.
    """
    esize = torch.tensor([], dtype=dtype).element_size()
    return 16 // math.gcd(16, esize)


def row_shape_supported(dtype: torch.dtype, N: int) -> bool:
    """Host-side check: can the TMA kernel handle an N-element row?

    chunk_bytes is chosen at compile time as ``min(row_bytes, 512)``, and
    ``cute.copy``'s transfer size is tied to the tensor shape, so every
    chunk must be full-sized. That means ``row_bytes`` must be a multiple
    of ``chunk_bytes`` (always true when row_bytes < 512) and
    ``chunk_bytes`` must itself be 16-byte aligned.
    """
    esize = torch.tensor([], dtype=dtype).element_size()
    row_bytes = N * esize
    chunk_bytes = min(row_bytes, _MAX_CHUNK_BYTES)
    if chunk_bytes % 16 != 0:
        return False
    return row_bytes % chunk_bytes == 0


def _plan_grid(M: int, D: int, chunk_elems: int, sm: int) -> tuple[int, int, int]:
    """Pick ``(grid_x, grid_y, d_tile_size)``.

    Strategy: keep the classic 1D schedule (grid_y=1, whole D per warp with
    internal double-buffering) whenever the row-axis alone saturates the
    GPU. When N is too small for that, split D across the y-axis so every
    SM gets work. The y-tile size is ``chunk_elems`` so warps don't lose
    the within-warp pipeline completely, but several chunks' worth of
    D-tiles go to different CTAs in parallel.
    """
    # 1 warp per CTA: need many more CTAs than an 8-warp layout to keep
    # occupancy up. sm*32 target with a sm*64 clamp works well across
    # uniform / high_cont / few_idx on B200.
    row_ctas = M
    target_ctas = sm * 32
    if row_ctas >= target_ctas:
        grid_x = min(row_ctas, sm * 64)
        return grid_x, 1, D
    # Split D across y until we hit the target, but don't shrink tiles
    # below chunk_elems (that would break the within-warp pipeline).
    n_d_tiles = (D + chunk_elems - 1) // chunk_elems
    want_y = max(1, target_ctas // max(row_ctas, 1))
    grid_y = min(n_d_tiles, want_y)
    # d_tile_size rounded up to chunk_elems multiples; last tile may be
    # shorter (handled by d_end clamp in the kernel).
    tiles_per_y = (n_d_tiles + grid_y - 1) // grid_y
    d_tile_size = tiles_per_y * chunk_elems
    # Recompute grid_y now that each y-slot holds tiles_per_y chunks.
    grid_y = (D + d_tile_size - 1) // d_tile_size
    grid_x = row_ctas
    return grid_x, grid_y, d_tile_size


def tma_scatter_add_into(
    out: torch.Tensor,
    index_1d: torch.Tensor,
    src: torch.Tensor,
) -> None:
    """In-place: ``out[index_1d[i], :] += src[i, :]`` for every i.

    ``out`` and ``src`` are 2D (M_out, N) and (M_src, N) with inner-dim
    stride 1; the outer row stride can differ from N (e.g. a slice of a
    wider buffer). ``index_1d`` is 1D int64 of length M_src. N must
    satisfy ``row_shape_supported``; the cond checks this.
    """
    M, N = src.shape
    chunk_elems = _chunk_elems_for(src.dtype, N)
    contig = src.stride(0) == N and out.stride(0) == N
    compiled = _compile_tma_scatter(src.dtype, N, contig)
    sm = torch.cuda.get_device_properties(out.device).multi_processor_count

    grid_x, grid_y, d_tile_size = _plan_grid(M, N, chunk_elems, sm)
    compiled(
        src,
        index_1d,
        out,
        M,
        N,
        d_tile_size,
        grid_x,
        grid_y,
        src.stride(0),
        out.stride(0),
    )
