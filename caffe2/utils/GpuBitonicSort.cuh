#ifndef CAFFE2_UTILS_GPU_BITONIC_SORT_H_
#define CAFFE2_UTILS_GPU_BITONIC_SORT_H_

#include "caffe2/utils/math.h"
#include "caffe2/utils/GpuDefs.cuh"

namespace caffe2 {

/// The maximum in-block bitonic sort we support
constexpr int kMaxBitonicSortSize = 4096;

template <typename T>
__device__ inline void swapVars(T& t1, T& t2) {
  T tmp = t1;
  t1 = t2;
  t2 = tmp;
}

template <typename Comparator, typename K, typename V>
__device__ inline void bitonicSwap(K& kA, V& vA,
                                   K& kB, V& vB,
                                   bool dir,
                                   const Comparator& comp) {
  bool swap = comp(kA, vA, kB, vB);
  if (swap == dir) {
    swapVars(kA, kB);
    swapVars(vA, vB);
  }
};

template <typename Comparator, typename K, typename V,
          int Power2SortSize,
          int ThreadsPerBlock>
__device__ inline void bitonicSort(K* keys,
                                   V* values,
                                   const Comparator& comp) {
  static_assert(Power2SortSize <= kMaxBitonicSortSize,
                "sort size <= 4096 only supported");
  // Assume the sort is taking place in shared memory
  // static_assert(Power2SortSize * (sizeof(K) + sizeof(V)) < 32768,
  //               "sort data too large (>32768 bytes)");
  static_assert(math::integerIsPowerOf2(Power2SortSize),
                "sort size must be power of 2");
  static_assert(math::integerIsPowerOf2(ThreadsPerBlock),
                "threads in block must be power of 2");

  // If what we are sorting is too small, then not all threads
  // participate
  constexpr int numThreadsForSort = Power2SortSize / 2;
  constexpr bool allThreads = numThreadsForSort >= ThreadsPerBlock;

  // If what we are sorting is too large, then threads must loop more
  // than once
  constexpr int loopPerThread =
    allThreads ? numThreadsForSort / ThreadsPerBlock : 1;

#pragma unroll
  for (int size = 2; size < Power2SortSize; size *= 2) {

#pragma unroll
    for (int stride = size / 2; stride > 0; stride /= 2) {

#pragma unroll
      for (int loop = 0; loop < loopPerThread; ++loop) {
        int threadId = loop * ThreadsPerBlock + threadIdx.x;
        bool flag = ((threadId & (size / 2)) != 0);

        int pos = 2 * threadId - (threadId & (stride - 1));

        if (allThreads || (threadId < numThreadsForSort)) {
          bitonicSwap<Comparator, K, V>(
            keys[pos], values[pos],
            keys[pos + stride], values[pos + stride],
            flag, comp);
        }

        __syncthreads();
      }
    }
  }

#pragma unroll
  for (int stride = Power2SortSize / 2; stride > 0; stride /= 2) {

#pragma unroll
    for (int loop = 0; loop < loopPerThread; ++loop) {
      int threadId = loop * ThreadsPerBlock + threadIdx.x;

      int pos = 2 * threadId - (threadId & (stride - 1));

      if (allThreads || (threadId < numThreadsForSort)) {
        bitonicSwap<Comparator, K, V>(
          keys[pos], values[pos],
          keys[pos + stride], values[pos + stride],
          false, comp);
      }

      __syncthreads();
    }
  }
}

template <typename Comparator, typename K, typename V, int Power2SortSize>
__device__ inline void warpBitonicSort(K* keys,
                                       V* values,
                                       const Comparator& comp) {
  // Smaller sorts should use a warp shuffle sort
  static_assert(Power2SortSize > kWarpSize,
                "sort not large enough");
  static_assert(math::integerIsPowerOf2(Power2SortSize),
                "sort size must be power of 2");
  static_assert(Power2SortSize <= kMaxBitonicSortSize,
                "sort size <= 4096 only supported");

  // If what we are sorting is too large, then lanes must loop more
  // than once
  constexpr int loopPerThread = (Power2SortSize / 2) / kWarpSize;
  int laneId = getLaneId();

#pragma unroll
  for (int size = 2; size < Power2SortSize; size *= 2) {

#pragma unroll
    for (int stride = size / 2; stride > 0; stride /= 2) {

#pragma unroll
      for (int loop = 0; loop < loopPerThread; ++loop) {
        int threadId = loop * kWarpSize + laneId;
        bool flag = ((threadId & (size / 2)) != 0);

        int pos = 2 * threadId - (threadId & (stride - 1));

        bitonicSwap<Comparator, K, V>(
          keys[pos], values[pos],
          keys[pos + stride], values[pos + stride],
          flag, comp);

        __threadfence_block();
      }
    }
  }

#pragma unroll
  for (int stride = Power2SortSize / 2; stride > 0; stride /= 2) {

#pragma unroll
    for (int loop = 0; loop < loopPerThread; ++loop) {
      int threadId = loop * kWarpSize + laneId;

      int pos = 2 * threadId - (threadId & (stride - 1));

      bitonicSwap<Comparator, K, V>(
        keys[pos], values[pos],
        keys[pos + stride], values[pos + stride],
        false, comp);

      __threadfence_block();
    }
  }
}


}  // namespace caffe2

#endif  // CAFFE2_UTILS_GPU_BITONIC_SORT_H_
