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
#if CUDA_VERSION >= 9000
  T vote = __ballot_sync(__activemask(), in);
#else
  T vote = __ballot(in);
#endif

  T index = __popc(getLaneMaskLe() & vote);
  T carry = __popc(vote);

  int warp = threadIdx.x / 32;

  // Per each warp, write out a value
  if (getLaneId() == 0) {
    smem[warp] = carry;
  }

  __syncthreads();

  // Sum across warps in one thread. This appears to be faster than a
  // warp shuffle scan for CC 3.0+
  if (threadIdx.x == 0) {
    int current = 0;
    for (int i = 0; i < blockDim.x / 32; ++i) {
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
  *carry = smem[(blockDim.x / 32) - 1];

  if (KillWARDependency) {
    __syncthreads();
  }
}

}  // namespace caffe2

#endif  // CAFFE2_UTILS_GPU_SCAN_UTILS_H_
