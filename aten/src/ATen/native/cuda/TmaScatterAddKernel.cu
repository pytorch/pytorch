// TMA-based scatter_add using cp.reduce.async.bulk (sm_90+)
// Separated into its own file to avoid include conflicts between
// <cuda/ptx> and PyTorch headers (Loops.cuh etc).

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ceil_div.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#if !defined(USE_ROCM)
#include <cuda/ptx>
#endif

// Must use ::cuda::ptx because inside namespace at::native,
// unqualified cuda:: resolves to the sibling at::cuda namespace.
namespace at::native {

template <typename scalar_t, typename index_t>
__global__ void tma_scatter_add_kernel(
    scalar_t* __restrict__ self_data,
    const scalar_t* __restrict__ src_data,
    const index_t* __restrict__ idx,
    int num_ind, int D, int64_t self_dim_size,
    int64_t self_stride, int64_t src_stride,
    int threads_per_entry, int entries_per_block, int chunk_elems) {
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    namespace ptx = ::cuda::ptx;

    extern __shared__ char smem_raw[];

    int entry_in_block = threadIdx.x / threads_per_entry;
    int lane = threadIdx.x - entry_in_block * threads_per_entry;

    // smem layout: [data region: entries_per_block * 2 * chunk_elems scalars]
    //              [mbarrier region: entries_per_block * 2 uint64_t, 8-byte aligned]
    // Mbarrier memory must never alias with data targeted by TMA operations.
    int buf_elems = 2 * chunk_elems;
    int data_region_bytes = entries_per_block * buf_elems * static_cast<int>(sizeof(scalar_t));
    int mbar_offset = (data_region_bytes + 7) & ~7;

    scalar_t* buf0 = reinterpret_cast<scalar_t*>(smem_raw) + entry_in_block * buf_elems;
    uint64_t* mbar0 = reinterpret_cast<uint64_t*>(smem_raw + mbar_offset) + entry_in_block * 2;
    uint64_t* mbar1 = mbar0 + 1;
    uint64_t* mbars[2] = {mbar0, mbar1};

    if (lane == 0) {
        ptx::mbarrier_init(mbar0, 1u);
        ptx::mbarrier_init(mbar1, 1u);
    }
    __syncwarp();

    int mbar_phase[2] = {0, 0};

    for (int base = blockIdx.x * entries_per_block; base < num_ind;
         base += gridDim.x * entries_per_block) {

        int entry_id = base + entry_in_block;
        if (entry_id >= num_ind) continue;

        int64_t ind = idx[entry_id];
        CUDA_KERNEL_ASSERT_VERBOSE(ind >= 0 && ind < self_dim_size && "tma scatter add index out of bounds",
            "Expected 0 <= index < self_dim_size(%ld), but got index = %ld", self_dim_size, ind);

        const scalar_t* src_entry = src_data + static_cast<int64_t>(entry_id) * src_stride;
        scalar_t* dst_entry = self_data + ind * self_stride;

        int phase = 0;
        for (int off = 0; off < D; off += chunk_elems, phase++) {
            int cur = phase & 1;
            int cur_elems = min(chunk_elems, D - off);
            uint32_t cur_bytes = cur_elems * sizeof(scalar_t);

            if (phase >= 2 && lane == 0) {
                ptx::cp_async_bulk_wait_group(ptx::n32_t<1>{});
            }
            __syncwarp();

            if (lane == 0) {
                ptx::mbarrier_arrive_expect_tx(
                    ptx::sem_release, ptx::scope_cta, ptx::space_shared,
                    mbars[cur], cur_bytes);
                ptx::cp_async_bulk(
                    ptx::space_shared, ptx::space_global,
                    buf0 + cur * chunk_elems, src_entry + off, cur_bytes, mbars[cur]);
            }
            while (!ptx::mbarrier_try_wait_parity(
                mbars[cur], static_cast<uint32_t>(mbar_phase[cur] & 1))) {}
            mbar_phase[cur]++;

            if (lane == 0) {
                using cuda_type = std::conditional_t<
                    std::is_same_v<scalar_t, c10::Half>, __half,
                    std::conditional_t<std::is_same_v<scalar_t, c10::BFloat16>, __nv_bfloat16, scalar_t>>;
                ptx::cp_reduce_async_bulk(
                    ptx::space_global, ptx::space_shared, ptx::op_add,
                    reinterpret_cast<cuda_type*>(dst_entry + off),
                    reinterpret_cast<cuda_type*>(buf0 + cur * chunk_elems),
                    cur_bytes);
                ptx::cp_async_bulk_commit_group();
            }
        }

        if (lane == 0) {
            ptx::cp_async_bulk_wait_group(ptx::n32_t<0>{});
        }
        __syncwarp();
    }

#else
    CUDA_KERNEL_ASSERT(false && "tma_scatter_add_kernel requires sm_90+");
#endif
}

template <typename scalar_t, typename index_t>
void tma_scatter_add_kernel_launch(
    scalar_t* self_data, const scalar_t* src_data, index_t* idx, int num_ind,
    int D, int64_t self_dim_size,
    int64_t self_stride_bytes, int64_t src_stride_bytes) {
#if !defined(USE_ROCM)
    constexpr int max_threads = 256;
    constexpr int threads_per_entry = 32;
    int chunk_elems = std::min(D, static_cast<int>(512 / sizeof(scalar_t)));

    int entries_per_block = max_threads / threads_per_entry;
    int block_size = entries_per_block * threads_per_entry;

    int buf_elems = 2 * chunk_elems;
    int data_region_bytes = entries_per_block * buf_elems * static_cast<int>(sizeof(scalar_t));
    int mbar_offset = (data_region_bytes + 7) & ~7;
    int smem = mbar_offset + entries_per_block * 2 * static_cast<int>(sizeof(uint64_t));

    int64_t self_stride = self_stride_bytes / sizeof(scalar_t);
    int64_t src_stride = src_stride_bytes / sizeof(scalar_t);

    auto mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int grid = std::min(at::ceil_div(num_ind, entries_per_block), mpc * 8);

    tma_scatter_add_kernel<scalar_t, index_t>
        <<<grid, block_size, smem, at::cuda::getCurrentCUDAStream()>>>(
        self_data, src_data, idx, num_ind, D, self_dim_size,
        self_stride, src_stride,
        threads_per_entry, entries_per_block, chunk_elems);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
    TORCH_CHECK(false, "TMA scatter_add not supported on ROCm");
#endif
}

#define INSTANTIATE_TMA_SCATTER_ADD(scalar_t) \
template void tma_scatter_add_kernel_launch<scalar_t, int64_t>( \
    scalar_t*, const scalar_t*, int64_t*, int, int, int64_t, int64_t, int64_t); \
template void tma_scatter_add_kernel_launch<scalar_t, int32_t>( \
    scalar_t*, const scalar_t*, int32_t*, int, int, int64_t, int64_t, int64_t);

INSTANTIATE_TMA_SCATTER_ADD(float)
INSTANTIATE_TMA_SCATTER_ADD(double)
INSTANTIATE_TMA_SCATTER_ADD(c10::Half)
INSTANTIATE_TMA_SCATTER_ADD(c10::BFloat16)
#undef INSTANTIATE_TMA_SCATTER_ADD

} // namespace at::native
