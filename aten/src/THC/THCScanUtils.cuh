#ifndef THC_SCAN_UTILS_INC
#define THC_SCAN_UTILS_INC

#include <THC/THCAsmUtils.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <c10/macros/Macros.h>

// Collection of in-kernel scan / prefix sum utilities


// Extends the above Inclusive Scan to support segments. It has the same properties
// but also takes a flag array that indicates the starts of "segments", i.e. individual
// units to scan. For example, consider the following (+)-scan that is segmented:
//
// Input:  [1, 3, 2, 4, 1, 2, 3, 2, 1, 4]
// Flags:  [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
// Output:  1  4  6  4  5  2  3  5  1  5
//
// So we see that each "flag" resets the scan to that index.
template <typename T, class BinaryOp, int Power2ScanSize>
__device__ void segmentedInclusivePrefixScan(T *smem, bool *bmem, BinaryOp binop) {
  // Reduce step ("upsweep")
#pragma unroll
  for (int stride = 1; stride < Power2ScanSize; stride <<= 1) {
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < Power2ScanSize) {
      smem[index] = bmem[index] ? smem[index] : binop(smem[index], smem[index - stride]);
      bmem[index] = bmem[index] | bmem[index - stride];
    }
    __syncthreads();
  }

  // Post-reduce step ("downsweep")
#pragma unroll
  for (int stride = Power2ScanSize / 4; stride > 0; stride >>= 1) {
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if ((index + stride) < Power2ScanSize) {
      smem[index + stride] = bmem[index + stride] ? smem[index + stride] : binop(smem[index + stride], smem[index]);
      bmem[index + stride] = bmem[index + stride] | bmem[index];
    }
    __syncthreads();
  }
}

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
#if defined (__HIP_PLATFORM_HCC__)
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
  *carry = smem[THCCeilDiv<int>(blockDim.x, C10_WARP_SIZE) - 1];

  if (KillWARDependency) {
    __syncthreads();
  }
}

#endif // THC_SCAN_UTILS_INC
