// TMA-based scatter reduce using cp.reduce.async.bulk (sm_90+, CUDA 12.8+)
// Supports add (f32, f64, f16, bf16) and min/max (s32, s64, f16, bf16).
// Uses inline PTX rather than cuda::ptx wrappers to avoid CCCL version
// compatibility issues across different CUDA toolkit versions.

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ceil_div.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

namespace at::native {

enum class TmaReduceOp : int { ADD = 0, MIN = 1, MAX = 2 };

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
static int tma_scatter_single_chunk_rows_per_warp(
    int row_bytes, int warps_per_block, int num_ind) {
    int rows_per_warp = row_bytes <= 512 ? 16 : 1;
    int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int64_t min_blocks = 8L * num_sms;
    while (rows_per_warp > 1) {
        int64_t blocks =
            at::ceil_div(num_ind, warps_per_block * rows_per_warp);
        if (blocks >= min_blocks) {
            break;
        }
        rows_per_warp /= 2;
    }
    return rows_per_warp;
}
#endif

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

// --- bulk_reduce_add ---

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

// --- bulk_reduce_min ---

template <typename scalar_t>
__device__ __forceinline__ void bulk_reduce_min(void* global_dst, const void* smem_src, uint32_t size);

template <>
__device__ __forceinline__ void bulk_reduce_min<int32_t>(void* dst, const void* src, uint32_t size) {
    uint64_t src_addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(src_addr) : "l"(reinterpret_cast<uint64_t>(src)));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s32 [%0], [%1], %2;"
        : : "l"(dst), "l"(src_addr), "r"(size) : "memory");
}

template <>
__device__ __forceinline__ void bulk_reduce_min<int64_t>(void* dst, const void* src, uint32_t size) {
    uint64_t src_addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(src_addr) : "l"(reinterpret_cast<uint64_t>(src)));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s64 [%0], [%1], %2;"
        : : "l"(dst), "l"(src_addr), "r"(size) : "memory");
}

template <>
__device__ __forceinline__ void bulk_reduce_min<c10::Half>(void* dst, const void* src, uint32_t size) {
    uint64_t src_addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(src_addr) : "l"(reinterpret_cast<uint64_t>(src)));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.f16 [%0], [%1], %2;"
        : : "l"(dst), "l"(src_addr), "r"(size) : "memory");
}

template <>
__device__ __forceinline__ void bulk_reduce_min<c10::BFloat16>(void* dst, const void* src, uint32_t size) {
    uint64_t src_addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(src_addr) : "l"(reinterpret_cast<uint64_t>(src)));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.bf16 [%0], [%1], %2;"
        : : "l"(dst), "l"(src_addr), "r"(size) : "memory");
}

// --- bulk_reduce_max ---

template <typename scalar_t>
__device__ __forceinline__ void bulk_reduce_max(void* global_dst, const void* smem_src, uint32_t size);

template <>
__device__ __forceinline__ void bulk_reduce_max<int32_t>(void* dst, const void* src, uint32_t size) {
    uint64_t src_addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(src_addr) : "l"(reinterpret_cast<uint64_t>(src)));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s32 [%0], [%1], %2;"
        : : "l"(dst), "l"(src_addr), "r"(size) : "memory");
}

template <>
__device__ __forceinline__ void bulk_reduce_max<int64_t>(void* dst, const void* src, uint32_t size) {
    uint64_t src_addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(src_addr) : "l"(reinterpret_cast<uint64_t>(src)));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s64 [%0], [%1], %2;"
        : : "l"(dst), "l"(src_addr), "r"(size) : "memory");
}

template <>
__device__ __forceinline__ void bulk_reduce_max<c10::Half>(void* dst, const void* src, uint32_t size) {
    uint64_t src_addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(src_addr) : "l"(reinterpret_cast<uint64_t>(src)));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.f16 [%0], [%1], %2;"
        : : "l"(dst), "l"(src_addr), "r"(size) : "memory");
}

template <>
__device__ __forceinline__ void bulk_reduce_max<c10::BFloat16>(void* dst, const void* src, uint32_t size) {
    uint64_t src_addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(src_addr) : "l"(reinterpret_cast<uint64_t>(src)));
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.bf16 [%0], [%1], %2;"
        : : "l"(dst), "l"(src_addr), "r"(size) : "memory");
}

// --- compile-time dispatch ---

template <typename scalar_t, TmaReduceOp op>
__device__ __forceinline__ void bulk_reduce(void* dst, const void* src, uint32_t size) {
    if constexpr (op == TmaReduceOp::ADD) {
        bulk_reduce_add<scalar_t>(dst, src, size);
    } else if constexpr (op == TmaReduceOp::MIN) {
        bulk_reduce_min<scalar_t>(dst, src, size);
    } else {
        bulk_reduce_max<scalar_t>(dst, src, size);
    }
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

template <typename scalar_t, typename index_t, TmaReduceOp op, bool multiple_rows_per_warp>
__global__ void tma_scatter_reduce_kernel(
    scalar_t* __restrict__ self_data,
    const scalar_t* __restrict__ src_data,
    const index_t* __restrict__ idx,
    int num_ind, int D, int64_t self_dim_size,
    int64_t self_stride, int64_t src_stride,
    int chunk_elems, int rows_per_warp) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080 && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

    extern __shared__ char smem_raw[];

    constexpr int threads_per_entry = C10_WARP_SIZE;
    int warp_in_block = threadIdx.x / threads_per_entry;
    int lane = threadIdx.x - warp_in_block * threads_per_entry;
    int warps_per_block = blockDim.x / threads_per_entry;
    int entries_per_block = warps_per_block * rows_per_warp;

    int buf_elems = 2 * chunk_elems;
    int data_region_bytes = warps_per_block * buf_elems * static_cast<int>(sizeof(scalar_t));
    int mbar_offset = (data_region_bytes + 7) & ~7;

    scalar_t* buf0 = reinterpret_cast<scalar_t*>(smem_raw) + warp_in_block * buf_elems;
    uint64_t* mbar0 = reinterpret_cast<uint64_t*>(smem_raw + mbar_offset) + warp_in_block * 2;
    uint64_t* mbar1 = mbar0 + 1;
    uint64_t* mbars[2] = {mbar0, mbar1};

    if (lane == 0) {
        tma::mbar_init(mbar0, 1u);
        tma::mbar_init(mbar1, 1u);
    }
    __syncwarp();

    int mbar_phase[2] = {0, 0};
    int entry_base = blockIdx.x * entries_per_block + warp_in_block * rows_per_warp;
    if (entry_base >= num_ind) return;
    int global_phase = 0;

    if constexpr (multiple_rows_per_warp) {
        int num_rows = min(rows_per_warp, num_ind - entry_base);
        uint32_t cur_bytes = D * sizeof(scalar_t);

        for (int row = 0; row < num_rows; row++, global_phase++) {
            int entry_id = entry_base + row;
            int cur = global_phase & 1;

            int64_t ind = idx[entry_id];
            CUDA_KERNEL_ASSERT_VERBOSE(ind >= 0 && ind < self_dim_size && "tma scatter reduce index out of bounds",
                "Expected 0 <= index < self_dim_size(%ld), but got index = %ld", self_dim_size, ind);

            const scalar_t* src_entry = src_data + static_cast<int64_t>(entry_id) * src_stride;
            scalar_t* dst_entry = self_data + ind * self_stride;

            if (global_phase >= 2 && lane == 0) {
                tma::wait_group_lt1();
            }
            __syncwarp();

            if (lane == 0) {
                tma::mbar_expect_tx(mbars[cur], cur_bytes);
                tma::bulk_load(buf0 + cur * chunk_elems, src_entry, cur_bytes, mbars[cur]);
            }
            while (!tma::mbar_try_wait_parity(
                mbars[cur], static_cast<uint32_t>(mbar_phase[cur] & 1))) {}
            mbar_phase[cur]++;

            if (lane == 0) {
                tma::bulk_reduce<scalar_t, op>(dst_entry, buf0 + cur * chunk_elems, cur_bytes);
                tma::commit_group();
            }
        }
    } else {
        int entry_id = entry_base;

        int64_t ind = idx[entry_id];
        CUDA_KERNEL_ASSERT_VERBOSE(ind >= 0 && ind < self_dim_size && "tma scatter reduce index out of bounds",
            "Expected 0 <= index < self_dim_size(%ld), but got index = %ld", self_dim_size, ind);

        const scalar_t* src_entry = src_data + static_cast<int64_t>(entry_id) * src_stride;
        scalar_t* dst_entry = self_data + ind * self_stride;

        for (int off = blockIdx.y * chunk_elems; off < D;
             off += gridDim.y * chunk_elems, global_phase++) {
            int cur = global_phase & 1;
            int cur_elems = min(chunk_elems, D - off);
            uint32_t cur_bytes = cur_elems * sizeof(scalar_t);

            if (global_phase >= 2 && lane == 0) {
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
                tma::bulk_reduce<scalar_t, op>(dst_entry + off, buf0 + cur * chunk_elems, cur_bytes);
                tma::commit_group();
            }
        }
    }

    if (lane == 0) {
        tma::wait_group_lt0();
    }
    __syncwarp();

#else
    CUDA_KERNEL_ASSERT(false && "tma_scatter_reduce_kernel requires sm_90+");
#endif
}

template <typename scalar_t, typename index_t, TmaReduceOp op>
static void tma_scatter_reduce_launch_impl(
    scalar_t* self_data, const scalar_t* src_data, index_t* idx, int num_ind,
    int D, int64_t self_dim_size,
    int64_t self_stride_bytes, int64_t src_stride_bytes) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
    constexpr int max_threads = 256;
    constexpr int threads_per_entry = C10_WARP_SIZE;
    constexpr int warps_per_block = max_threads / threads_per_entry;
    int chunk_elems = std::min(D, static_cast<int>(512 / sizeof(scalar_t)));
    int num_chunks = at::ceil_div(D, chunk_elems);

    int64_t self_stride = self_stride_bytes / sizeof(scalar_t);
    int64_t src_stride = src_stride_bytes / sizeof(scalar_t);

    int rows_per_warp = 1;
    if (num_chunks == 1) {
        int row_bytes = D * static_cast<int>(sizeof(scalar_t));
        rows_per_warp = tma_scatter_single_chunk_rows_per_warp(
            row_bytes, warps_per_block, num_ind);
    }

    TORCH_INTERNAL_ASSERT(rows_per_warp == 1 || num_chunks == 1,
        "multi-row requires single chunk");
    TORCH_INTERNAL_ASSERT(rows_per_warp == 1 || chunk_elems == D,
        "multi-row requires chunk_elems == D");

    int entries_per_block = warps_per_block * rows_per_warp;
    int grid_x = at::ceil_div(num_ind, entries_per_block);
    constexpr int min_chunks_per_block = 4;
    uint32_t grid_y = std::min(
        static_cast<uint32_t>(std::max(1, num_chunks / min_chunks_per_block)),
        static_cast<uint32_t>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1]));

    int data_region_bytes = warps_per_block * 2 * chunk_elems * static_cast<int>(sizeof(scalar_t));
    int mbar_offset = (data_region_bytes + 7) & ~7;
    int smem = mbar_offset + warps_per_block * 2 * static_cast<int>(sizeof(uint64_t));

    dim3 grid = {static_cast<uint32_t>(grid_x), grid_y, 1};

    if (rows_per_warp > 1) {
        tma_scatter_reduce_kernel<scalar_t, index_t, op, true>
            <<<grid, max_threads, smem, at::cuda::getCurrentCUDAStream()>>>(
            self_data, src_data, idx, num_ind, D, self_dim_size,
            self_stride, src_stride,
            chunk_elems, rows_per_warp);
    } else {
        tma_scatter_reduce_kernel<scalar_t, index_t, op, false>
            <<<grid, max_threads, smem, at::cuda::getCurrentCUDAStream()>>>(
            self_data, src_data, idx, num_ind, D, self_dim_size,
            self_stride, src_stride,
            chunk_elems, rows_per_warp);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
    TORCH_CHECK(false, "TMA scatter reduce requires CUDA 12.8+ and NVIDIA GPU");
#endif
}

// scatter_add entry point (preserves existing API)
template <typename scalar_t, typename index_t>
void tma_scatter_add_kernel_launch(
    scalar_t* self_data, const scalar_t* src_data, index_t* idx, int num_ind,
    int D, int64_t self_dim_size,
    int64_t self_stride_bytes, int64_t src_stride_bytes) {
    tma_scatter_reduce_launch_impl<scalar_t, index_t, TmaReduceOp::ADD>(
        self_data, src_data, idx, num_ind, D, self_dim_size,
        self_stride_bytes, src_stride_bytes);
}

// scatter_reduce min/max entry point
template <typename scalar_t, typename index_t>
void tma_scatter_reduce_kernel_launch(
    scalar_t* self_data, const scalar_t* src_data, index_t* idx, int num_ind,
    int D, int64_t self_dim_size,
    int64_t self_stride_bytes, int64_t src_stride_bytes, bool is_min) {
    if (is_min) {
        tma_scatter_reduce_launch_impl<scalar_t, index_t, TmaReduceOp::MIN>(
            self_data, src_data, idx, num_ind, D, self_dim_size,
            self_stride_bytes, src_stride_bytes);
    } else {
        tma_scatter_reduce_launch_impl<scalar_t, index_t, TmaReduceOp::MAX>(
            self_data, src_data, idx, num_ind, D, self_dim_size,
            self_stride_bytes, src_stride_bytes);
    }
}

// scatter_add instantiations: f32, f64, f16, bf16
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

// scatter_reduce min/max instantiations: s32, s64, f16, bf16
#define INSTANTIATE_TMA_SCATTER_REDUCE(scalar_t) \
template void tma_scatter_reduce_kernel_launch<scalar_t, int64_t>( \
    scalar_t*, const scalar_t*, int64_t*, int, int, int64_t, int64_t, int64_t, bool); \
template void tma_scatter_reduce_kernel_launch<scalar_t, int32_t>( \
    scalar_t*, const scalar_t*, int32_t*, int, int, int64_t, int64_t, int64_t, bool);

INSTANTIATE_TMA_SCATTER_REDUCE(int32_t)
INSTANTIATE_TMA_SCATTER_REDUCE(int64_t)
INSTANTIATE_TMA_SCATTER_REDUCE(c10::Half)
INSTANTIATE_TMA_SCATTER_REDUCE(c10::BFloat16)
#undef INSTANTIATE_TMA_SCATTER_REDUCE

} // namespace at::native
