#pragma once

#include <ATen/ATen.h>

// Contents of this file are copied from THCUNN/common.h for the ease of porting
// THCUNN functions into ATen.

namespace at { namespace cuda { namespace detail {

// CUDA: grid stride looping
//
// int64_t _i_n_d_e_x specifically prevents overflow in the loop increment.
// Overflow in the cast is not checked in order to avoid another conditional.
// Array sizes should be checked before the kernel is launched.
#define CUDA_KERNEL_LOOP(i, n) \
  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;                                \
  for (int i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)

// Use 1024 threads per block, which requires cuda sm_2x or above
constexpr int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
  AT_ASSERTM(N > 0, "CUDA kernel launch blocks must be positive, but got N=", N);
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

}}}  // namespace at::cuda::detail
