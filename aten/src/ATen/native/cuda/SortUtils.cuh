#pragma once
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>

#include <ATen/cuda/cub.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/Sort.h>
#include <ATen/native/StridedRandomAccessor.h>

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

template <int KeyDims, int ValueDims,
          int block_size, int items_per_thread,
          typename K, typename V, typename IndexType>
C10_LAUNCH_BOUNDS_1(block_size)
__global__ void
radixSortKVInPlace(at::cuda::detail::TensorInfo<K, IndexType> keys,
                   IndexType keySlices,
                   IndexType keySliceSize,
                   IndexType keySliceStride,
                   at::cuda::detail::TensorInfo<V, IndexType> values,
                   IndexType valueSliceStride,
                   bool descending) {
  static_assert(block_size > 0, "");

  // Find the slice of the tensor that we are sorting
  const IndexType linearIndex = getLinearBlockId<IndexType>();
  // Tiling the slices could have us be out of bounds, if there are a
  // lot of slices to sort
  if (linearIndex >= keySlices) {
    return;
  }

  const IndexType keyStartOffset =
    at::cuda::detail::IndexToOffset<K, IndexType, KeyDims>::get(linearIndex, keys);
  const IndexType valueStartOffset =
    at::cuda::detail::IndexToOffset<V, IndexType, ValueDims>::get(linearIndex, values);

  K *keys_slice = &keys.data[keyStartOffset];
  V *values_slice = &values.data[valueStartOffset];

  StridedRandomAccessor<K, IndexType> keys_iter(keys_slice, keySliceStride);
  StridedRandomAccessor<V, IndexType> values_iter(values_slice, valueSliceStride);

  namespace cub = ROCM_HIPCUB(at_cuda_detail::cub);

  using key_t = typename at::cuda::cub::detail::cuda_type<K>::type;
  using LoadKeys = cub::BlockLoad<K, block_size, items_per_thread,
                                  cub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE>;
  using LoadValues = cub::BlockLoad<V, block_size, items_per_thread,
                                    cub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE>;
  using Sort = cub::BlockRadixSort<key_t, block_size, items_per_thread, V>;
  using StoreKeys = cub::BlockStore<K, block_size, items_per_thread,
                                    cub::BLOCK_STORE_TRANSPOSE>;
  using StoreValues = cub::BlockStore<V, block_size, items_per_thread,
                                      cub::BLOCK_STORE_TRANSPOSE>;

  __shared__ union {
    typename LoadKeys::TempStorage load_keys;
    typename LoadValues::TempStorage load_values;
    typename Sort::TempStorage sort;
    typename StoreKeys::TempStorage store_keys;
    typename StoreValues::TempStorage store_values;
  } tmp_storage;

  // cub's Block operations operate on a fixed number of items, but the
  // actual slice we are sorting might be smaller. So, we need to make
  // up the difference with keys that will always sort higher.
  const K invalid_key = [descending] {
    using radix_t = typename cub::Traits<key_t>::UnsignedBits;
    union {
      K key;
      radix_t radix;
    } tmp;
    tmp.radix = descending ?
        cub::Traits<key_t>::LOWEST_KEY :
        cub::Traits<key_t>::MAX_KEY;
    return tmp.key;
  }();
  const V invalid_value = static_cast<V>(0);

  // Load inputs
  K local_keys[items_per_thread];
  V local_values[items_per_thread];

  LoadKeys(tmp_storage.load_keys).Load(keys_iter, local_keys, keySliceSize, invalid_key);
  __syncthreads();
  LoadValues(tmp_storage.load_values).Load(values_iter, local_values, keySliceSize, invalid_value);
  __syncthreads();

  // Sort!
  if (descending) {
    Sort(tmp_storage.sort).SortDescending(
        reinterpret_cast<key_t (&)[items_per_thread]>(local_keys),
        local_values);
  } else {
    Sort(tmp_storage.sort).Sort(
        reinterpret_cast<key_t (&)[items_per_thread]>(local_keys),
        local_values);
  }
  __syncthreads();

  // Store outputs
  StoreKeys(tmp_storage.store_keys).Store(keys_iter, local_keys, keySliceSize);
  __syncthreads();
  StoreValues(tmp_storage.store_values).Store(values_iter, local_values, keySliceSize);
}

}} // at::native
