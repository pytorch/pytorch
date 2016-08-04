#ifndef THC_SCAN_UTILS_INC
#define THC_SCAN_UTILS_INC

#include "THCAsmUtils.cuh"

// Collection of in-kernel scan / prefix sum utilities

// Inclusive prefix sum using shared memory
template <typename T, bool KillWARDependency>
__device__ void inclusivePrefixSum(T* smem, T in, T* out) {
  // FIXME: this is a slow, simple implementation; need up/down sweep,
  // prevent smem conflicts
  smem[threadIdx.x] = in;

  __syncthreads();

  for (int offset = 1; offset < blockDim.x; offset *= 2) {
    T val = 0;

    if (threadIdx.x >= offset) {
      val = smem[threadIdx.x - offset] + smem[threadIdx.x];
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
template <typename T, bool KillWARDependency>
__device__ void exclusivePrefixSum(T* smem, T in, T* out, T* carry) {
  // FIXME: crappy implementation
  // We kill write-after-read dependencies separately below, hence the `false`
  inclusivePrefixSum<T, false>(smem, in, out);

  *out -= in;
  *carry = smem[blockDim.x - 1];

  // Prevent write-after-read dependencies on smem usage above if necessary
  if (KillWARDependency) {
    __syncthreads();
  }
}

// Inclusive prefix sum for binary vars using intra-warp voting +
// shared memory
template <typename T, bool KillWARDependency>
__device__ void inclusiveBinaryPrefixSum(T* smem, bool in, T* out) {
  // Within-warp, we use warp voting.
  T vote = __ballot(in);
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
      smem[i] += current;
      current += v;
    }
  }

  __syncthreads();

  // load the carry from the preceding warp
  if (warp >= 1) {
    index += smem[warp - 1];
  }

  *out = index;

  if (KillWARDependency) {
    __syncthreads();
  }
}

// Exclusive prefix sum for binary vars using intra-warp voting +
// shared memory
template <typename T, bool KillWARDependency>
__device__ void exclusiveBinaryPrefixSum(T* smem, bool in, T* out, T* carry) {
  inclusiveBinaryPrefixSum<T, false>(smem, in, out);

  // Inclusive to exclusive
  *out -= (T) in;

  // The outgoing carry for all threads is the last warp's sum
  *carry = smem[(blockDim.x / 32) - 1];

  if (KillWARDependency) {
    __syncthreads();
  }
}

#endif // THC_SCAN_UTILS_INC
