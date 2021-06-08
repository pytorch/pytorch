#pragma once

#include <ATen/cuda/DeviceUtils.cuh>

namespace at {
namespace native {
namespace cuda_utils {

constexpr int kCUDABlockReduceNumThreads = 512;

// Sums `val` accross all threads in a warp.
//
// Assumptions:
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

// Sums `val` accross all threads in a block.
//
// Assumptions:
//   - Thread blocks are an 1D set of threads (indexed with `threadIdx.x` only)
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
//   - `shared` should be a pointer to shared memory with size of, at least,
//     `sizeof(T) * number_of_warps`
template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < blockDim.x / C10_WARP_SIZE) ? shared[lid] : 0;
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

} // namespace cuda_utils
} // namespace native
} // namespace at
