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

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
#include <cuda.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <ATen/cuda/detail/LazyNVRTC.h>
#endif

namespace at::native {

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080

template <typename scalar_t>
struct TmaDataType;

template <>
struct TmaDataType<float> {
    static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
};
template <>
struct TmaDataType<double> {
    static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
};
template <>
struct TmaDataType<c10::Half> {
    static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
};
template <>
struct TmaDataType<c10::BFloat16> {
    static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
};

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

template <int N>
__device__ __forceinline__ void wait_group_lt() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}

__device__ __forceinline__ void wait_group_lt1() {
    wait_group_lt<1>();
}

__device__ __forceinline__ void wait_group_lt0() {
    wait_group_lt<0>();
}

__device__ __forceinline__ void bulk_tensor_load_2d(
    void* smem_dst, const void* tensor_map,
    int32_t col_coord, int32_t row_coord, uint64_t* mbar) {
    uint64_t dst_addr, mbar_addr;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;"
        : "=l"(dst_addr) : "l"(reinterpret_cast<uint64_t>(smem_dst)));
    asm volatile("cvta.to.shared::cta.u64 %0, %1;"
        : "=l"(mbar_addr) : "l"(reinterpret_cast<uint64_t>(mbar)));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        : : "r"(static_cast<uint32_t>(dst_addr)),
            "l"(tensor_map),
            "r"(col_coord), "r"(row_coord),
            "r"(static_cast<uint32_t>(mbar_addr)) : "memory");
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

// 2D TMA tile kernel for small-N (num_chunks == 1).
// Each pipeline stage loads K rows with a single cp.async.bulk.tensor.2d
// instruction, then issues K separate bulk-reduces (destinations are scattered).
// Producer-ahead-of-consumer pipeline overlaps the current load with the
// previous stage's reduces. 2D TMA handles non-contiguous src (D < stride)
// and OOB tiles (last tile with < K rows) via hardware zero-fill.
template <typename scalar_t, typename index_t, int NUM_STAGES>
__global__ void tma_scatter_add_kernel_2d(
    scalar_t* __restrict__ self_data,
    const scalar_t* __restrict__ src_data,
    const index_t* __restrict__ idx,
    int num_ind, int D, int64_t self_dim_size,
    int64_t self_stride, int64_t src_stride,
    const __grid_constant__ CUtensorMap tma_desc) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080 && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

    constexpr int K = 4;

    extern __shared__ char smem_raw[];

    int warp_id = threadIdx.x / C10_WARP_SIZE;
    int lane_id = threadIdx.x % C10_WARP_SIZE;
    int warps_per_block = blockDim.x / C10_WARP_SIZE;

    int tile_elems = K * D;
    int slab_elems = tile_elems * NUM_STAGES;
    int data_region_bytes = warps_per_block * slab_elems * static_cast<int>(sizeof(scalar_t));
    int mbar_offset = (data_region_bytes + 7) & ~7;

    scalar_t* my_buf = reinterpret_cast<scalar_t*>(smem_raw) + warp_id * slab_elems;
    uint64_t* mbars = reinterpret_cast<uint64_t*>(smem_raw + mbar_offset) + warp_id * NUM_STAGES;

    if (lane_id == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            tma::mbar_init(&mbars[s], 1u);
        }
    }
    __syncwarp();

    uint32_t tile_bytes = K * D * sizeof(scalar_t);

    auto drain_tile = [&](int tile, int stage, int& parity) {
        while (!tma::mbar_try_wait_parity(
            &mbars[stage],
            static_cast<uint32_t>(parity))) {}
        if (stage == NUM_STAGES - 1) {
            parity ^= 1;
        }

        if (lane_id == 0) {
            int base_row = tile * K;
            scalar_t* buf = my_buf + stage * tile_elems;
            for (int k = 0; k < K; k++) {
                int row = base_row + k;
                if (row < num_ind) {
                    int64_t ind = idx[row];
                    CUDA_KERNEL_ASSERT_VERBOSE(
                        ind >= 0 && ind < self_dim_size &&
                        "tma scatter add index out of bounds",
                        "Expected 0 <= index < self_dim_size(%ld), "
                        "but got index = %ld", self_dim_size, ind);
                    scalar_t* dst_row = self_data + ind * self_stride;
                    tma::bulk_reduce_add<scalar_t>(
                        dst_row, buf + k * D,
                        D * sizeof(scalar_t));
                }
            }
            tma::commit_group();
        }
    };

    int n_tiles = at::ceil_div(num_ind, K);
    int first_tile = blockIdx.x * warps_per_block + warp_id;
    int warp_stride = gridDim.x * warps_per_block;

    auto issue_load = [&](int tile, int stage) {
        if (lane_id == 0) {
            tma::mbar_expect_tx(&mbars[stage], tile_bytes);
            tma::bulk_tensor_load_2d(
                my_buf + stage * tile_elems,
                &tma_desc,
                0, static_cast<int32_t>(tile * K),
                &mbars[stage]);
        }
    };

    if (first_tile >= n_tiles) {
        return;
    }

    // Circular-buffered pipeline with NUM_STAGES smem buffers.
    // Prolog issues 1 load. Steady-state issues 1 load + 1 drain per
    // iteration; the load is always 1 step ahead of the drain, cycling
    // through stages 0..NUM_STAGES-1. A stage is reused NUM_STAGES
    // iterations after its drain, so wait_group_lt<NUM_STAGES-2> ensures
    // the async reduce from that drain has finished reading the buffer.
    int tile = first_tile;
    int phase = 0;
    int parity = 0;

    // Prolog: issue first load
    issue_load(tile, 0);
    tile += warp_stride;
    phase++;

    // Steady-state: load next tile, drain previous tile
    for (; tile < n_tiles; tile += warp_stride, phase++) {
        int load_stage = phase % NUM_STAGES;

        if (phase >= NUM_STAGES && lane_id == 0) {
            tma::wait_group_lt<NUM_STAGES - 2>();
        }
        __syncwarp();

        issue_load(tile, load_stage);
        drain_tile(tile - warp_stride, (phase - 1) % NUM_STAGES, parity);
    }

    // Epilog: drain last tile, wait for all reduces
    drain_tile(tile - warp_stride, (phase - 1) % NUM_STAGES, parity);
    if (lane_id == 0) {
        tma::wait_group_lt0();
    }
    __syncwarp();

#else
    CUDA_KERNEL_ASSERT(false && "tma_scatter_add_kernel_2d requires sm_90+");
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

    int64_t self_stride = self_stride_bytes / sizeof(scalar_t);
    int64_t src_stride = src_stride_bytes / sizeof(scalar_t);

    if (num_chunks == 1) {
        constexpr int K = 4;
        constexpr int NUM_STAGES = 3;
        int tile_elems = K * D;
        int slab_elems = tile_elems * NUM_STAGES;
        int data_region_bytes = entries_per_block * slab_elems *
            static_cast<int>(sizeof(scalar_t));
        int mbar_offset = (data_region_bytes + 7) & ~7;
        int smem = mbar_offset + entries_per_block * NUM_STAGES *
            static_cast<int>(sizeof(uint64_t));

        int n_tiles = at::ceil_div(num_ind, K);
        int grid_x = at::ceil_div(n_tiles, entries_per_block);
        int num_sms = at::cuda::getCurrentDeviceProperties()
            ->multiProcessorCount;
        grid_x = std::min(grid_x, num_sms * 32);

        alignas(128) CUtensorMap desc;
        cuuint64_t globalDims[2] = {
            static_cast<cuuint64_t>(D),
            static_cast<cuuint64_t>(num_ind)};
        cuuint64_t globalStrides[1] = {
            static_cast<cuuint64_t>(src_stride_bytes)};
        cuuint32_t boxDims[2] = {
            static_cast<cuuint32_t>(D),
            static_cast<cuuint32_t>(K)};
        cuuint32_t elemStrides[2] = {1, 1};

        CUresult res = at::cuda::detail::lazyNVRTC.cuTensorMapEncodeTiled(
            &desc,
            TmaDataType<scalar_t>::value,
            2,
            const_cast<scalar_t*>(src_data),
            globalDims,
            globalStrides,
            boxDims,
            elemStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
        TORCH_INTERNAL_ASSERT(res == CUDA_SUCCESS,
            "cuTensorMapEncodeTiled failed: ", static_cast<int>(res));

        dim3 grid = {static_cast<uint32_t>(grid_x), 1, 1};
        int block_size = entries_per_block * threads_per_entry;

        auto kernel = tma_scatter_add_kernel_2d<scalar_t, index_t, NUM_STAGES>;
        if (smem > static_cast<int>(at::cuda::getCurrentDeviceProperties()
                ->sharedMemPerBlock)) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        }
        kernel<<<grid, block_size, smem, at::cuda::getCurrentCUDAStream()>>>(
            self_data, src_data, idx, num_ind, D, self_dim_size,
            self_stride, src_stride, desc);
    } else {
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

        dim3 grid = {static_cast<uint32_t>(grid_x), grid_y, 1};

        tma_scatter_add_kernel<scalar_t, index_t>
            <<<grid, block_size, smem, at::cuda::getCurrentCUDAStream()>>>(
            self_data, src_data, idx, num_ind, D, self_dim_size,
            self_stride, src_stride,
            entries_per_block, chunk_elems);
    }
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
