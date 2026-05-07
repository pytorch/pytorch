"""TMA-based scatter_add for the ``index.unsqueeze(-1).expand(-1, N)`` pattern.

Port of the CUDA C++ kernel from https://github.com/pytorch/pytorch/pull/182675
to CuTeDSL. Each warp handles one source row (within an assigned D-range):
a TMA bulk load stages ``src[i, d_start:d_end]`` into smem, then
``cp.reduce.async.bulk.global.shared::cta.bulk_group.add`` deposits the
reduction into ``out[index[i], d_start:d_end]``.

Chunks along D in 512-byte slices, double-buffered so the next chunk's
load overlaps the current chunk's reduce.

Grid layout is 2D to maintain SM utilization for shapes with small N and
large D: ``(grid_x, grid_y)``, where ``grid_y`` partitions the D axis
into disjoint chunks and each warp iterates only its assigned chunk.
When N is large the host sets ``grid_y = 1`` so the inner chunk loop
recovers the original schedule (double-buffered all of D within each
warp). When N is tiny (say 100 rows, D=640K), ``grid_y`` grows so the
grid still fills the machine.

Restrictions (enforced by the host cond in ``cutedsl_impl.py``):
  - sm_90+ (cp.reduce.async.bulk availability)
  - dim == 0, rank >= 2, self/src contiguous
  - index is the expanded-1D pattern (same shape as src, stride 0 on every
    axis except 0)
  - dtype in {fp32, fp16, bf16}
  - N * elem_size % 16 == 0 (16-byte TMA alignment)
"""

import math

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, const_expr, Float16, Float32, Int32, Int64, Uint64

import torch
from torch._vendor.quack.cache_utils import jit_cache

from ._ptx import bulk_load, cvta_smem, make_bulk_reduce_add


# Tile size on the wire: 512 bytes per chunk.
_CHUNK_BYTES = 512
_WARPS_PER_BLOCK = 8
_THREADS_PER_BLOCK = 32 * _WARPS_PER_BLOCK


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


def _make_kernel(dtype, elem_bytes: int, chunk_elems: int, reduce_op, scale: bool):
    """Build a dtype-specialized kernel closure. dtype/elem_bytes/chunk_elems
    /scale are Python-time constants that the preprocessor folds at
    cute.compile time.

    When ``scale`` is True, after the TMA bulk load completes the 32 lanes
    cooperatively multiply the smem buffer by the runtime ``alpha`` before
    the bulk-reduce. The ``alpha == 1`` fast path uses ``scale=False`` and
    avoids the smem pass + proxy fence entirely.
    """

    @cute.kernel
    def _kernel(
        mSrc: cute.Tensor,
        mIndex: cute.Tensor,
        mOut: cute.Tensor,
        num_entries: Int32,
        D: Int32,
        d_tile_size: Int32,
        alpha: cute.Numeric,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdim_x, _, _ = cute.arch.grid_dim()

        warp_in_block = tidx // Int32(32)
        lane = tidx - warp_in_block * Int32(32)

        # Assigned D-range for this CTA. bidy partitions D into disjoint
        # slices of d_tile_size; grid_y == ceil(D / d_tile_size). When
        # d_tile_size == D (the large-N case), grid_y == 1 and every CTA
        # sees the whole D.
        d_start = bidy * d_tile_size
        d_end = d_start + d_tile_size
        if d_end > D:
            d_end = D

        smem = cutlass.utils.SmemAllocator()
        mbar_array = smem.allocate_array(Uint64, num_elems=_WARPS_PER_BLOCK * 2)
        # 128-byte alignment so each warp's sub-buffer is TMA-aligned.
        # allocate_tensor takes byte_alignment positionally; allocate_array
        # on cutedsl < 4.5 does not support the keyword.
        sAll = smem.allocate_tensor(
            dtype,
            cute.make_layout(_WARPS_PER_BLOCK * 2 * chunk_elems),
            128,
        )
        sAll_ptr = sAll.iterator

        warp_base = warp_in_block * Int32(2 * chunk_elems)
        buf0_ptr = sAll_ptr + warp_base
        buf1_ptr = sAll_ptr + (warp_base + Int32(chunk_elems))
        buf0_u64 = cvta_smem(buf0_ptr)
        buf1_u64 = cvta_smem(buf1_ptr)

        mbar0_ptr = mbar_array + warp_in_block * Int32(2)
        mbar1_ptr = mbar0_ptr + Int32(1)

        if lane == Int32(0):
            cute.arch.mbarrier_init(mbar0_ptr, 1)
            cute.arch.mbarrier_init(mbar1_ptr, 1)
        cute.arch.sync_threads()

        mbar0_phase = Int32(0)
        mbar1_phase = Int32(0)

        entries_per_block = Int32(_WARPS_PER_BLOCK)
        base = bidx * entries_per_block
        while base < num_entries:
            entry_id = base + warp_in_block
            if entry_id < num_entries:
                # aten passes int64 index. Read first element of the row
                # (all elements along axis 1 are identical: the cond
                # requires index.stride(1) == 0).
                idx_t = cute.make_tensor(
                    mIndex.iterator + Int64(entry_id) * Int64(mIndex.stride[0]),
                    cute.make_layout(1),
                )
                r = Int64(idx_t[0])

                # Iterate chunks within [d_start, d_end) with double-buffering.
                phase = Int32(0)
                off = d_start
                while off < d_end:
                    cur_elems = d_end - off
                    if cur_elems > Int32(chunk_elems):
                        cur_elems = Int32(chunk_elems)
                    cur_bytes = cur_elems * Int32(elem_bytes)

                    cur = phase & Int32(1)

                    if phase >= Int32(2) and lane == Int32(0):
                        cute.arch.cp_async_bulk_wait_group(1, read=False)

                    if lane == Int32(0):
                        if cur == Int32(0):
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar0_ptr, cur_bytes
                            )
                        else:
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar1_ptr, cur_bytes
                            )

                        src_off = Int64(entry_id) * Int64(D) + Int64(off)
                        gmem_src_u64 = Int64((mSrc.iterator + src_off).toint())

                        if cur == Int32(0):
                            bulk_load(
                                buf0_u64,
                                gmem_src_u64,
                                cur_bytes,
                                cvta_smem(mbar0_ptr),
                            )
                        else:
                            bulk_load(
                                buf1_u64,
                                gmem_src_u64,
                                cur_bytes,
                                cvta_smem(mbar1_ptr),
                            )

                    if cur == Int32(0):
                        cute.arch.mbarrier_wait(mbar0_ptr, mbar0_phase & Int32(1))
                        mbar0_phase = mbar0_phase + Int32(1)
                    else:
                        cute.arch.mbarrier_wait(mbar1_ptr, mbar1_phase & Int32(1))
                        mbar1_phase = mbar1_phase + Int32(1)

                    if const_expr(scale):
                        # All 32 lanes cooperatively scale smem in place.
                        # Warp is already synchronized by the mbarrier_wait.
                        # fence_view_async_shared releases these generic
                        # writes to the async proxy so the bulk-reduce
                        # sees them.
                        alpha_t = dtype(alpha)
                        buf_ptr = buf0_ptr
                        if cur != Int32(0):
                            buf_ptr = buf1_ptr
                        i = lane
                        while i < cur_elems:
                            slot = cute.make_tensor(buf_ptr + i, cute.make_layout(1))
                            slot[0] = slot[0] * alpha_t
                            i = i + Int32(32)
                        cute.arch.sync_warp()
                        cute.arch.fence_view_async_shared()

                    if lane == Int32(0):
                        dst_off = r * Int64(D) + Int64(off)
                        gmem_dst_u64 = Int64((mOut.iterator + dst_off).toint())
                        if cur == Int32(0):
                            reduce_op(gmem_dst_u64, buf0_u64, cur_bytes)
                        else:
                            reduce_op(gmem_dst_u64, buf1_u64, cur_bytes)
                        cute.arch.cp_async_bulk_commit_group()

                    off = off + Int32(chunk_elems)
                    phase = phase + Int32(1)

                if lane == Int32(0):
                    cute.arch.cp_async_bulk_wait_group(0, read=False)

            base = base + gdim_x * entries_per_block

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
        alpha: cute.Numeric,
    ):
        _kernel(mSrc, mIndex, mOut, num_entries, D, d_tile_size, alpha).launch(
            grid=[grid_x, grid_y, 1],
            block=[_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch


def _alpha_dtype_for(torch_dtype: torch.dtype):
    """Alpha is passed as fp32 across the ABI; tvm_ffi doesn't support
    fp16/bf16 scalar args. Kernel casts to src dtype inside."""
    return Float32


@jit_cache
def _compile_tma_scatter(torch_dtype: torch.dtype, N: int, scale: bool):
    dtype = _TORCH_TO_CUTE[torch_dtype]
    elem_bytes = dtype.width // 8
    chunk_elems = _CHUNK_BYTES // elem_bytes
    reduce_op = _reduce_op_for(dtype)
    launcher = _make_kernel(dtype, elem_bytes, chunk_elems, reduce_op, scale)

    mSrc_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), N), stride=(cute.sym_int64(), 1)
    )
    mIndex_fake = cute.runtime.make_fake_tensor(
        Int64, (cute.sym_int(),), stride=(cute.sym_int64(),)
    )
    mOut_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), N), stride=(cute.sym_int64(), 1)
    )
    alpha_dtype = _alpha_dtype_for(torch_dtype)
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
        alpha_dtype(0.0),
        options="--enable-tvm-ffi",
    )


def min_d_divisor_for(dtype: torch.dtype) -> int:
    """Smallest value D must be divisible by so ``cp.async.bulk`` transfers
    stay 16-byte-aligned: ``D * sizeof(dtype) % 16 == 0``, i.e.
    ``D % (16 / gcd(16, sizeof(dtype))) == 0``.
    """
    esize = torch.tensor([], dtype=dtype).element_size()
    return 16 // math.gcd(16, esize)


def _plan_grid(M: int, D: int, chunk_elems: int, sm: int) -> tuple[int, int, int]:
    """Pick ``(grid_x, grid_y, d_tile_size)``.

    Strategy: keep the classic 1D schedule (grid_y=1, whole D per warp with
    internal double-buffering) whenever the row-axis alone saturates the
    GPU. When N is too small for that, split D across the y-axis so every
    SM gets work. The y-tile size is ``chunk_elems`` so warps don't lose
    the within-warp pipeline completely, but several chunks' worth of
    D-tiles go to different CTAs in parallel.
    """
    row_ctas = (M + _WARPS_PER_BLOCK - 1) // _WARPS_PER_BLOCK
    target_ctas = sm * 4  # modest oversubscription per SM
    if row_ctas >= target_ctas:
        # Row axis alone saturates; keep the classic schedule.
        grid_x = min(row_ctas, sm * 8)
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
    alpha: float = 1.0,
) -> None:
    """In-place: ``out[index_1d[i], :] += alpha * src[i, :]`` for every i.

    ``out`` and ``src`` must be contiguous (M_out, N) and (M_src, N);
    ``index_1d`` is 1D int64 of length M_src. N must satisfy
    ``min_d_divisor_for(src.dtype)``; the cond checks this.

    ``alpha == 1.0`` dispatches to the non-scaling variant (no smem scale
    pass, no proxy fence, one less register pressure item).
    """
    M, N = src.shape
    elem_bytes = torch.tensor([], dtype=src.dtype).element_size()
    chunk_elems = _CHUNK_BYTES // elem_bytes
    scale = alpha != 1.0
    compiled = _compile_tma_scatter(src.dtype, N, scale)
    sm = torch.cuda.get_device_properties(out.device).multi_processor_count

    grid_x, grid_y, d_tile_size = _plan_grid(M, N, chunk_elems, sm)
    alpha_dtype = _alpha_dtype_for(src.dtype)
    compiled(
        src,
        index_1d,
        out,
        M,
        N,
        d_tile_size,
        grid_x,
        grid_y,
        alpha_dtype(float(alpha)),
    )
