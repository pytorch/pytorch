#pragma once

#include <thrust/tuple.h>

#include <ATen/cuda/DeviceUtils.cuh>

namespace at {
namespace native {
namespace cuda_utils {

constexpr int kCUDABlockReduceNumThreads = 512;

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

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

template <typename T>
__inline__ __device__ thrust::tuple<int64_t, T, T> WarpReduceMoments(
    int64_t m0,
    T m1,
    T m2) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    const int64_t m0_add = WARP_SHFL_DOWN(m0, offset);
    const T m1_add = WARP_SHFL_DOWN(m1, offset);
    const T m2_add = WARP_SHFL_DOWN(m2, offset);
    const int64_t n = m0 + m0_add;
    const T c1 = n == 0 ? T(0) : static_cast<T>(m0) / static_cast<T>(n);
    const T c2 = n == 0 ? T(0) : T(1) - c1;
    const T delta = m1_add - m1;
    m0 = n;
    m1 = c1 * m1 + c2 * m1_add;
    m2 += m2_add + delta * delta * c1 * c2 * static_cast<T>(n);
  }
  return thrust::make_tuple(m0, m1, m2);
}

template <typename T>
__inline__ __device__ thrust::tuple<int64_t, T, T> BlockReduceMoments(
    int64_t m0,
    T m1,
    T m2,
    int64_t* m0_shared,
    T* m1_shared,
    T* m2_shared) {
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  thrust::tie(m0, m1, m2) = WarpReduceMoments(m0, m1, m2);
  __syncthreads();
  if (lid == 0) {
    m0_shared[wid] = m0;
    m1_shared[wid] = m1;
    m2_shared[wid] = m2;
  }
  __syncthreads();
  const bool in_warp = (threadIdx.x < blockDim.x / C10_WARP_SIZE);
  m0 = in_warp ? m0_shared[lid] : 0;
  m1 = in_warp ? m1_shared[lid] : 0;
  m2 = in_warp ? m2_shared[lid] : 0;
  if (wid == 0) {
    thrust::tie(m0, m1, m2) = WarpReduceMoments(m0, m1, m2);
  }
  return thrust::make_tuple(m0, m1, m2);
}

} // namespace cuda_utils
} // namespace native
} // namespace at
