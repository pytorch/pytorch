// TMA-based scatter_add using cp.reduce.async.bulk (sm_90+, CUDA 12.8+)
// Uses inline PTX rather than cuda::ptx wrappers to avoid CCCL version
// compatibility issues across different CUDA toolkit versions.
// Requires CUDA 12.8+ for cp.async.bulk.shared::cta.global (PTX ISA 8.7).

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ceil_div.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

namespace at::native {

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080 && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

namespace tma {

__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t count) {
    uint64_t addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(addr) : "l"(reinterpret_cast<uint64_t>(mbar)));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" : : "l"(addr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* mbar, uint32_t bytes) {
    uint64_t addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(addr) : "l"(reinterpret_cast<uint64_t>(mbar)));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        : : "l"(addr), "r"(bytes) : "memory");
}

__device__ __forceinline__ bool mbar_try_wait_parity(uint64_t* mbar, uint32_t phase_parity) {
    uint64_t addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(addr) : "l"(reinterpret_cast<uint64_t>(mbar)));
    uint32_t wait_complete;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(wait_complete) : "l"(addr), "r"(phase_parity) : "memory");
    return static_cast<bool>(wait_complete);
}

__device__ __forceinline__ void bulk_load(void* smem_dst, const void* global_src,
                                           uint32_t size, uint64_t* mbar) {
    uint64_t dst_addr, mbar_addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(dst_addr) : "l"(reinterpret_cast<uint64_t>(smem_dst)));
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(mbar_addr) : "l"(reinterpret_cast<uint64_t>(mbar)));
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        : : "l"(dst_addr), "l"(global_src), "r"(size), "l"(mbar_addr) : "memory");
}

template <typename scalar_t>
__device__ __forceinline__ void bulk_reduce_add(void* global_dst, const void* smem_src, uint32_t size);

template <>
__device__ __forceinline__ void bulk_reduce_add<float>(void* dst, const void* src, uint32_t size) {
    uint64_t src_addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(src_addr) : "l"(reinterpret_cast<uint64_t>(src)));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
        : : "l"(dst), "l"(src_addr), "r"(size) : "memory");
}

template <>
__device__ __forceinline__ void bulk_reduce_add<double>(void* dst, const void* src, uint32_t size) {
    uint64_t src_addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(src_addr) : "l"(reinterpret_cast<uint64_t>(src)));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64 [%0], [%1], %2;"
        : : "l"(dst), "l"(src_addr), "r"(size) : "memory");
}

template <>
__device__ __forceinline__ void bulk_reduce_add<c10::Half>(void* dst, const void* src, uint32_t size) {
    uint64_t src_addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(src_addr) : "l"(reinterpret_cast<uint64_t>(src)));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [%0], [%1], %2;"
        : : "l"(dst), "l"(src_addr), "r"(size) : "memory");
}

template <>
__device__ __forceinline__ void bulk_reduce_add<c10::BFloat16>(void* dst, const void* src, uint32_t size) {
    uint64_t src_addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(src_addr) : "l"(reinterpret_cast<uint64_t>(src)));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [%0], [%1], %2;"
        : : "l"(dst), "l"(src_addr), "r"(size) : "memory");
}

__device__ __forceinline__ void commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void wait_group_lt1() {
    asm volatile("cp.async.bulk.wait_group 1;" ::: "memory");
}

__device__ __forceinline__ void wait_group_lt0() {
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

} // namespace tma

#endif // !USE_ROCM && __CUDA_ARCH__ >= 900

template <typename scalar_t, typename index_t>
__global__ void tma_scatter_add_kernel(
    scalar_t* __restrict__ self_data,
    const scalar_t* __restrict__ src_data,
    const index_t* __restrict__ idx,
    int num_ind, int D, int64_t self_dim_size,
    int64_t self_stride, int64_t src_stride,
    int entries_per_block, int chunk_elems) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080 && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

    extern __shared__ char smem_raw[];

    // One warp per entry: lane 0 issues TMA commands, __syncwarp() synchronizes.
    constexpr int threads_per_entry = C10_WARP_SIZE;
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
        tma::mbar_init(mbar0, 1u);
        tma::mbar_init(mbar1, 1u);
    }
    __syncwarp();

    int mbar_phase[2] = {0, 0};

    {
        int entry_id = blockIdx.x * entries_per_block + entry_in_block;
        if (entry_id >= num_ind) return;

        int64_t ind = idx[entry_id];
        CUDA_KERNEL_ASSERT_VERBOSE(ind >= 0 && ind < self_dim_size && "tma scatter add index out of bounds",
            "Expected 0 <= index < self_dim_size(%ld), but got index = %ld", self_dim_size, ind);

        const scalar_t* src_entry = src_data + static_cast<int64_t>(entry_id) * src_stride;
        scalar_t* dst_entry = self_data + ind * self_stride;

        int phase = 0;
        for (int off = blockIdx.y * chunk_elems; off < D;
             off += gridDim.y * chunk_elems, phase++) {
            int cur = phase & 1;
            int cur_elems = min(chunk_elems, D - off);
            uint32_t cur_bytes = cur_elems * sizeof(scalar_t);

            if (phase >= 2 && lane == 0) {
                tma::wait_group_lt1();
            }
            __syncwarp();

            if (lane == 0) {
                tma::mbar_expect_tx(mbars[cur], cur_bytes);
                tma::bulk_load(buf0 + cur * chunk_elems, src_entry + off, cur_bytes, mbars[cur]);
            }
            while (!tma::mbar_try_wait_parity(
                mbars[cur], static_cast<uint32_t>(mbar_phase[cur] & 1))) {}
            mbar_phase[cur]++;

            if (lane == 0) {
                tma::bulk_reduce_add<scalar_t>(dst_entry + off, buf0 + cur * chunk_elems, cur_bytes);
                tma::commit_group();
            }
        }

        if (lane == 0) {
            tma::wait_group_lt0();
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
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
    constexpr int max_threads = 256;
    // One warp per entry: lane 0 issues TMA commands, __syncwarp() synchronizes.
    constexpr int threads_per_entry = C10_WARP_SIZE;
    int chunk_elems = std::min(D, static_cast<int>(512 / sizeof(scalar_t)));
    int num_chunks = at::ceil_div(D, chunk_elems);

    int entries_per_block = max_threads / threads_per_entry;
    int grid_x = at::ceil_div(num_ind, entries_per_block);
    // Spread chunks across grid.y but keep at least 4 per block for pipeline benefit
    constexpr int min_chunks_per_block = 4;
    uint32_t grid_y = std::min(
        static_cast<uint32_t>(std::max(1, num_chunks / min_chunks_per_block)),
        static_cast<uint32_t>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1]));
    int block_size = entries_per_block * threads_per_entry;

    int buf_elems = 2 * chunk_elems;
    int data_region_bytes = entries_per_block * buf_elems * static_cast<int>(sizeof(scalar_t));
    int mbar_offset = (data_region_bytes + 7) & ~7;
    int smem = mbar_offset + entries_per_block * 2 * static_cast<int>(sizeof(uint64_t));

    int64_t self_stride = self_stride_bytes / sizeof(scalar_t);
    int64_t src_stride = src_stride_bytes / sizeof(scalar_t);

    dim3 grid = {static_cast<uint32_t>(grid_x), grid_y, 1};

    tma_scatter_add_kernel<scalar_t, index_t>
        <<<grid, block_size, smem, at::cuda::getCurrentCUDAStream()>>>(
        self_data, src_data, idx, num_ind, D, self_dim_size,
        self_stride, src_stride,
        entries_per_block, chunk_elems);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
    TORCH_CHECK(false, "TMA scatter_add requires CUDA 12.8+ and NVIDIA GPU");
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
