#pragma once

#include <ATen/ceil_div.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/cuda/AsmUtils.cuh>
#include <c10/macros/Macros.h>

// Collection of in-kernel scan / prefix sum utilities

namespace at {
namespace cuda {

// Inclusive prefix sum for binary vars using intra-warp voting +
// shared memory
template <typename T, bool KillWARDependency, class BinaryFunction>
__device__ void inclusiveBinaryPrefixScan(T* smem, bool in, T* out, BinaryFunction binop) {
  // Within-warp, we use warp voting.
#if defined (USE_ROCM)
  unsigned long long int vote = WARP_BALLOT(in);
  T index = __popcll(getLaneMaskLe() & vote);
  T carry = __popcll(vote);
#else
  T vote = WARP_BALLOT(in);
  T index = __popc(getLaneMaskLe() & vote);
  T carry = __popc(vote);
#endif

  int warp = threadIdx.x / C10_WARP_SIZE;

  // Per each warp, write out a value
  if (getLaneId() == 0) {
    smem[warp] = carry;
  }

  __syncthreads();

  // Sum across warps in one thread. This appears to be faster than a
  // warp shuffle scan for CC 3.0+
  if (threadIdx.x == 0) {
    int current = 0;
    for (int i = 0; i < blockDim.x / C10_WARP_SIZE; ++i) {
      T v = smem[i];
      smem[i] = binop(smem[i], current);
      current = binop(current, v);
    }
  }

  __syncthreads();

  // load the carry from the preceding warp
  if (warp >= 1) {
    index = binop(index, smem[warp - 1]);
  }

  *out = index;

  if (KillWARDependency) {
    __syncthreads();
  }
}

// Exclusive prefix sum for binary vars using intra-warp voting +
// shared memory
template <typename T, bool KillWARDependency, class BinaryFunction>
__device__ void exclusiveBinaryPrefixScan(T* smem, bool in, T* out, T* carry, BinaryFunction binop) {
  inclusiveBinaryPrefixScan<T, false, BinaryFunction>(smem, in, out, binop);

  // Inclusive to exclusive
  *out -= (T) in;

  // The outgoing carry for all threads is the last warp's sum
  *carry = smem[at::ceil_div<int>(blockDim.x, C10_WARP_SIZE) - 1];

  if (KillWARDependency) {
    __syncthreads();
  }
}

}}  // namespace at::cuda
