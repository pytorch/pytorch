#pragma once
// Contents of this file are copied from THCUNN/common.h for the ease of porting
// THCUNN functions into ATen.

namespace at { namespace cuda { namespace detail {

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
constexpr int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

}}}  // namespace at::cuda::detail
