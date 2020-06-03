#ifndef CAFFE2_UTILS_GPU_SCAN_UTILS_H_
#define CAFFE2_UTILS_GPU_SCAN_UTILS_H_

#include "caffe2/utils/GpuDefs.cuh"

namespace caffe2 {

// from the cutorch library; can probably be replaced with their CUB
// equivalents
// Collection of in-kernel scan / prefix sum utilities

// Inclusive prefix sum using shared memory
template <typename T, bool KillWARDependency, class BinaryFunction>
__device__ void inclusivePrefixScan(T* smem, T in, T* out, BinaryFunction binop) {
  // FIXME: this is a slow, simple implementation; need up/down sweep,
  // prevent smem conflicts
  smem[threadIdx.x] = in;

  __syncthreads();

  for (int offset = 1; offset < blockDim.x; offset *= 2) {
    T val = 0;

    if (threadIdx.x >= offset) {
      val = binop(smem[threadIdx.x - offset], smem[threadIdx.x]);
    }

    __syncthreads();
    if (threadIdx.x >= offset) {
      smem[threadIdx.x] = val;
    }

    __syncthreads();
  }

  *out = smem[threadIdx.x];

  // Prevent write-after-read dependencies on smem usage above if necessary
  if (KillWARDependency) {
    __syncthreads();
  }
}

// Exclusive prefix sum using shared memory
template <typename T, bool KillWARDependency, class BinaryFunction>
__device__ void exclusivePrefixScan(T* smem, T in, T* out, T* carry, BinaryFunction binop) {
  // FIXME: crappy implementation
  // We kill write-after-read dependencies separately below, hence the `false`
  inclusivePrefixScan<T, false, BinaryFunction>(smem, in, out, binop);

  *out -= in;
  *carry = smem[blockDim.x - 1];

  // Prevent write-after-read dependencies on smem usage above if necessary
  if (KillWARDependency) {
    __syncthreads();
  }
}

// Inclusive prefix sum for binary vars using intra-warp voting +
// shared memory
template <typename T, bool KillWARDependency, class BinaryFunction>
__device__ void inclusiveBinaryPrefixScan(T* smem, bool in, T* out, BinaryFunction binop) {
  // Within-warp, we use warp voting.
#if defined(__HIP_PLATFORM_HCC__)
  unsigned long long int vote = __ballot(in);

  T index = __popcll(getLaneMaskLe() & vote);
  T carry = __popcll(vote);
#else
  T vote = __ballot_sync(__activemask(), in);
  T index = __popc(getLaneMaskLe() & vote);
  T carry = __popc(vote);
#endif  // __HIP_PLATFORM_HCC__

  int warp = threadIdx.x / kWarpSize;

  // Per each warp, write out a value
  if (getLaneId() == 0) {
    smem[warp] = carry;
  }

  __syncthreads();

  // Sum across warps in one thread. This appears to be faster than a
  // warp shuffle scan for CC 3.0+
  if (threadIdx.x == 0) {
    int current = 0;
    for (int i = 0; i < blockDim.x / kWarpSize; ++i) {
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
#if defined(__HIP_PLATFORM_HCC__)
  *carry = smem[math::DivUp<int>(blockDim.x, kWarpSize) - 1];
#else
  *carry = smem[(blockDim.x / kWarpSize) - 1];
#endif  // __HIP_PLATFORM_HCC__

  if (KillWARDependency) {
    __syncthreads();
  }
}

}  // namespace caffe2

#endif  // CAFFE2_UTILS_GPU_SCAN_UTILS_H_
