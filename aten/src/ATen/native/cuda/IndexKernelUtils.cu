#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/cuda/Atomic.cuh>

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/ceil_div.h>

namespace at::native {
template <int Alignment, typename index_t>
__global__ void vectorized_gather_kernel(char * out, char * inp, index_t * idx, int num_ind, int64_t slice_size, int64_t ind_dim_size, int64_t inp_stride, int64_t out_stride, bool allow_neg_indices) {
    int64_t ind = idx[blockIdx.x];
    if (allow_neg_indices) {
        ind = (ind < 0) ? ind + ind_dim_size : ind;
    }
    CUDA_KERNEL_ASSERT_VERBOSE(ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds", "Expected 0 <= index < ind_dim_size(%ld), but got index = %ld", ind_dim_size, ind);
    // off is guaranteed to be within int32 limits
    for (int32_t off = (blockDim.x * blockIdx.y + threadIdx.x) * Alignment; off < slice_size; off += blockDim.x * gridDim.y * Alignment) {
      auto vec = at::native::memory::ld_vec<Alignment>(inp + ind * inp_stride + off);
      at::native::memory::st_vec<Alignment>(out + blockIdx.x * (int32_t)out_stride + off, vec);  // out offset is guaranteed to be within int32 limits
    }
}



template <int64_t Alignment, typename index_t>
void vectorized_gather_kernel_launch(char * out, char * inp, index_t * idx, int num_ind,
                                     int64_t slice_size_in_bytes, int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes, bool allow_neg_indices){

  constexpr int64_t max_num_threads=256;
  auto num_threads = at::round_up(
      at::ceil_div(slice_size_in_bytes, Alignment),
      static_cast<int64_t>(at::cuda::warp_size()));
  uint32_t grid_y = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
  grid_y = std::min(static_cast<uint32_t>(at::ceil_div(slice_size_in_bytes, max_num_threads * Alignment)), grid_y);
  dim3 grid = {static_cast<uint32_t>(num_ind), grid_y, 1};
  auto block = std::min(max_num_threads, num_threads);
  vectorized_gather_kernel<Alignment, index_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, idx, num_ind, slice_size_in_bytes,
  ind_dim_size, inp_stride_bytes, out_stride_bytes, allow_neg_indices);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// explicit template instantiation
template void vectorized_gather_kernel_launch<16, int64_t>(char * out, char * inp, int64_t * idx, int num_ind, int64_t slice_size_in_bytes,
int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes, bool allow_neg_indices);
template void vectorized_gather_kernel_launch<16, int32_t>(char * out, char * inp, int32_t * idx, int num_ind, int64_t slice_size_in_bytes,
int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes, bool allow_neg_indices);

// Vectorized scatter_add: load a vector from src, atomicAdd each element to self
template <int Alignment, typename scalar_t>
__device__ __forceinline__ void atomicAddVec(
    scalar_t* dst, const at::native::memory::Vec<Alignment>& vec) {
  constexpr int N = Alignment / sizeof(scalar_t);

  if constexpr (std::is_same_v<scalar_t, float>) {
    #pragma unroll
    for (int i = 0; i < N; i++) {
      atomicAdd(dst + i, vec.f32[i]);
    }
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    const double* vals = reinterpret_cast<const double*>(&vec);
    #pragma unroll
    for (int i = 0; i < N; i++) {
      atomicAdd(dst + i, vals[i]);
    }
  } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700))
    const c10::Half* vals = reinterpret_cast<const c10::Half*>(&vec);
    #pragma unroll
    for (int i = 0; i < N; i++) {
      gpuAtomicAddNoReturn(dst + i, vals[i]);
    }
#else
    __half2* dst2 = reinterpret_cast<__half2*>(dst);
    const __half2* src2 = reinterpret_cast<const __half2*>(&vec);
    #pragma unroll
    for (int i = 0; i < N / 2; i++) {
      atomicAdd(dst2 + i, src2[i]);
    }
#endif
  } else if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
    const c10::BFloat16* vals = reinterpret_cast<const c10::BFloat16*>(&vec);
    #pragma unroll
    for (int i = 0; i < N; i++) {
      gpuAtomicAddNoReturn(dst + i, vals[i]);
    }
#else
    __nv_bfloat162* dst2 = reinterpret_cast<__nv_bfloat162*>(dst);
    const __nv_bfloat162* src2 = reinterpret_cast<const __nv_bfloat162*>(&vec);
    #pragma unroll
    for (int i = 0; i < N / 2; i++) {
      atomicAdd(dst2 + i, src2[i]);
    }
#endif
  }
}

template <int Alignment, typename scalar_t, typename index_t>
__global__ void vectorized_scatter_add_kernel(
    scalar_t* __restrict__ self_data,
    const scalar_t* __restrict__ src_data,
    const index_t* __restrict__ idx,
    int num_ind,
    int64_t slice_size_bytes,
    int64_t self_dim_size,
    int64_t self_stride_bytes,
    int64_t src_stride_bytes,
    int threads_per_entry,
    int entries_per_block) {

  int entry_in_block = threadIdx.x / threads_per_entry;
  int lane = threadIdx.x - entry_in_block * threads_per_entry;
  int entry_id = blockIdx.x * entries_per_block + entry_in_block;

  if (entry_id >= num_ind) return;

  int64_t ind = idx[entry_id];
  CUDA_KERNEL_ASSERT_VERBOSE(ind >= 0 && ind < self_dim_size && "vectorized scatter add kernel index out of bounds",
      "Expected 0 <= index < self_dim_size(%ld), but got index = %ld", self_dim_size, ind);

  scalar_t* dst_slice = reinterpret_cast<scalar_t*>(
      reinterpret_cast<char*>(self_data) + ind * self_stride_bytes);

  for (int32_t off = (blockIdx.y * threads_per_entry + lane) * Alignment;
       off < slice_size_bytes;
       off += gridDim.y * threads_per_entry * Alignment) {

    auto vec = at::native::memory::ld_vec<Alignment>(
        reinterpret_cast<const char*>(src_data) + entry_id * static_cast<int32_t>(src_stride_bytes) + off);

    scalar_t* dst = reinterpret_cast<scalar_t*>(
        reinterpret_cast<char*>(dst_slice) + off);

    atomicAddVec<Alignment>(dst, vec);
  }
}

template <int64_t Alignment, typename scalar_t, typename index_t>
void vectorized_scatter_add_kernel_launch(
    scalar_t* self_data, const scalar_t* src_data, index_t* idx, int num_ind,
    int64_t slice_size_in_bytes, int64_t self_dim_size,
    int64_t self_stride_bytes, int64_t src_stride_bytes) {

  constexpr int64_t max_num_threads = 256;
  int64_t num_vectors = slice_size_in_bytes / Alignment;
  int64_t threads_per_entry = std::min(
      at::round_up(num_vectors, static_cast<int64_t>(at::cuda::warp_size())),
      max_num_threads);
  int64_t entries_per_block = max_num_threads / threads_per_entry;
  int64_t block_size = entries_per_block * threads_per_entry;
  int64_t grid_x = at::ceil_div(static_cast<int64_t>(num_ind), entries_per_block);
  uint32_t grid_y = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
  grid_y = std::min(
      static_cast<uint32_t>(at::ceil_div(num_vectors, threads_per_entry)),
      grid_y);

  dim3 grid = {static_cast<uint32_t>(grid_x), grid_y, 1};
  vectorized_scatter_add_kernel<Alignment, scalar_t, index_t>
      <<<grid, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
      self_data, src_data, idx, num_ind, slice_size_in_bytes,
      self_dim_size, self_stride_bytes, src_stride_bytes,
      static_cast<int>(threads_per_entry), static_cast<int>(entries_per_block));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define INSTANTIATE_SCATTER_ADD(scalar_t) \
template void vectorized_scatter_add_kernel_launch<16, scalar_t, int64_t>( \
    scalar_t*, const scalar_t*, int64_t*, int, int64_t, int64_t, int64_t, int64_t); \
template void vectorized_scatter_add_kernel_launch<16, scalar_t, int32_t>( \
    scalar_t*, const scalar_t*, int32_t*, int, int64_t, int64_t, int64_t, int64_t);

INSTANTIATE_SCATTER_ADD(float)
INSTANTIATE_SCATTER_ADD(double)
INSTANTIATE_SCATTER_ADD(c10::Half)
INSTANTIATE_SCATTER_ADD(c10::BFloat16)
#undef INSTANTIATE_SCATTER_ADD


}
