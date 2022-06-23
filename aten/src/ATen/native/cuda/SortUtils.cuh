#pragma once
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>

#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/Sort.h>

namespace at { namespace native {

template <typename T>
__device__ inline void swapVars(T& t1, T& t2) {
  T tmp = t1;
  t1 = t2;
  t2 = tmp;
}

template <typename Comparator, typename K, typename V>
__device__ inline void bitonicSwap(K& kA, V& vA, bool& validA,
                                   K& kB, V& vB, bool& validB,
                                   bool dir,
                                   const Comparator& comp) {
  // Invalid entries always sort to the end
  bool swap = (comp(kA, kB) && validA) || !validB;
  if (swap == dir) {
    swapVars(kA, kB);
    swapVars(vA, vB);
    swapVars(validA, validB);
  }
};

template <int Power2SortSize, typename IndexType, typename Comparator,
          typename K, typename V>
__device__ inline void bitonicSort(K *keys,
                                   V *values,
                                   bool *valid,
                                   const Comparator& comp) {
#if !defined(USE_ROCM)
#pragma unroll
#endif
  for (unsigned int size = 2; size < Power2SortSize; size *= 2) {
    bool flag = ((threadIdx.x & (size / 2)) != 0);

#if !defined(USE_ROCM)
#pragma unroll
#endif
    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {

      __syncthreads();

      unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      bitonicSwap<Comparator, K, V>(
        keys[pos], values[pos], valid[pos],
        keys[pos + stride], values[pos + stride], valid[pos + stride],
        flag, comp);
    }
  }

#if !defined(USE_ROCM)
#pragma unroll
#endif
  for (unsigned int stride = Power2SortSize / 2; stride > 0; stride /= 2) {

    __syncthreads();

    unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    bitonicSwap<Comparator, K, V>(
      keys[pos], values[pos], valid[pos],
      keys[pos + stride], values[pos + stride], valid[pos + stride],
      false, comp);
  }

  __syncthreads();

}

// at::cuda::detail::TensorInfo version
// Sorts (key, value) pairs (in different tensors) in-place; i.e.,
// modifies the input `keys` and `values`
template <int KeyDims, int ValueDims, int block_dim_x, int max_block_dim_y,
          typename K, typename V, typename Comparator, typename IndexType>
C10_LAUNCH_BOUNDS_1(block_dim_x * max_block_dim_y)
__global__ void
bitonicSortKVInPlace(at::cuda::detail::TensorInfo<K, IndexType> keys,
                     IndexType keySlices,
                     IndexType keySliceSize,
                     IndexType keySliceStride,
                     at::cuda::detail::TensorInfo<V, IndexType> values,
                     IndexType valueSliceStride,
                     Comparator comp) {
  // Find the slice of the tensor that we are sorting
  // NOTE: blockDim.y may be less max_block_dim_y
  const IndexType blockIndex = getLinearBlockId<IndexType>();
  const IndexType linearIndex = blockIndex * blockDim.y + threadIdx.y;

  // If the entire block is out of bounds exit early
  if (blockIndex * blockDim.y >= keySlices) {
    return;
  }
  // It's also possible for some rows of a block to be out of bounds
  // but all thread need to run for __syncthreads to work.
  const bool row_valid = linearIndex < keySlices;

  constexpr int items_per_thread = 2;
  constexpr int Power2SortSize = block_dim_x * items_per_thread;

  // Storage for max_block_dim_y sorts performed in parallel
  __shared__ K blockSharedKeys[max_block_dim_y][Power2SortSize];
  __shared__ V blockSharedValues[max_block_dim_y][Power2SortSize];
  __shared__ bool blockSharedValid[max_block_dim_y][Power2SortSize];

  auto sharedKeys = blockSharedKeys[threadIdx.y];
  auto sharedValues = blockSharedValues[threadIdx.y];
  auto sharedValid = blockSharedValid[threadIdx.y];

  const IndexType keyStartOffset =
    at::cuda::detail::IndexToOffset<K, IndexType, KeyDims>::get(linearIndex, keys);
  const IndexType valueStartOffset =
    at::cuda::detail::IndexToOffset<V, IndexType, ValueDims>::get(linearIndex, values);

  // Load 2 values per thread into the shared workspace
  #pragma unroll
  for (int k = 0; k < items_per_thread; ++k) {
    auto idx = threadIdx.x + k * blockDim.x;
    bool valid = row_valid && idx < keySliceSize;

    sharedKeys[idx] = valid ?
        keys.data[idx * keySliceStride + keyStartOffset] : K{};
    sharedValues[idx] = valid ?
        values.data[idx * valueSliceStride + valueStartOffset] : V{};
    sharedValid[idx] = valid;
  }

  // Sort!
  bitonicSort<Power2SortSize, IndexType>(
      sharedKeys, sharedValues, sharedValid, comp);

  if (!row_valid) {
    return;
  }

  // Store outputs
  #pragma unroll
  for (int k = 0; k < items_per_thread; ++k) {
    auto idx = threadIdx.x + k * blockDim.x;
    if (idx < keySliceSize) {
      keys.data[idx * keySliceStride + keyStartOffset] = sharedKeys[idx];
      values.data[idx * valueSliceStride + valueStartOffset] = sharedValues[idx];
    }
  }
}

}} // at::native
