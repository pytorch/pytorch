#pragma once
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>

#include <ATen/ATen.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Resize.h>
#include <THC/THCNumerics.cuh> // for ScalarConvert
#include <THC/THCSortUtils.cuh>

namespace at { namespace native {

// at::cuda::detail::TensorInfo version
// Sorts (key, value) pairs (in different tensors) in-place; i.e.,
// modifies the input `keys` and `values`
template <typename K, typename V,
          int KeyDims, int ValueDims,
          typename Comparator, typename IndexType, int Power2SortSize>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void
bitonicSortKVInPlace(at::cuda::detail::TensorInfo<K, IndexType> keys,
                     IndexType keySlices,
                     IndexType keySliceSize,
                     IndexType keySliceStride,
                     at::cuda::detail::TensorInfo<V, IndexType> values,
                     IndexType valueSliceStride,
                     Comparator comp) {
  // Find the slice of the tensor that we are sorting
  const IndexType linearIndex = getLinearBlockId<IndexType>();
  // Tiling the slices could have us be out of bounds, if there are a
  // lot of slices to sort
  if (linearIndex >= keySlices) {
    return;
  }

  __shared__ K sharedKeys[Power2SortSize];
  __shared__ V sharedValues[Power2SortSize];
  __shared__ bool sharedValid[Power2SortSize];

  const IndexType keyStartOffset =
    at::cuda::detail::IndexToOffset<K, IndexType, KeyDims>::get(linearIndex, keys);
  const IndexType valueStartOffset =
    at::cuda::detail::IndexToOffset<V, IndexType, ValueDims>::get(linearIndex, values);

  // If the sort size is 1, the data is already sorted
  if (Power2SortSize == 1) {
    return;
  } else {
    // Otherwise, each thread is responsible for loading and storing 2
    // elements. The sort size is guaranteed to be >= 2
    const int elem1 = threadIdx.x;
    const int elem2 = threadIdx.x + (Power2SortSize / 2);

    bool valid1 = (elem1 < keySliceSize);
    K k1 = valid1 ?
      keys.data[keyStartOffset + elem1 * keySliceStride] : ScalarConvert<int, K>::to(0);
    V v1 = valid1 ?
      values.data[valueStartOffset + elem1 * valueSliceStride] : ScalarConvert<int, V>::to(0);

    sharedKeys[elem1] = k1;
    sharedValues[elem1] = v1;
    sharedValid[elem1] = valid1;

    bool valid2 = (elem2 < keySliceSize);
    K k2 = valid2 ?
      keys.data[keyStartOffset + elem2 * keySliceStride] : ScalarConvert<int, K>::to(0);
    V v2 = valid2 ?
      values.data[valueStartOffset + elem2 * valueSliceStride] : ScalarConvert<int, V>::to(0);

    sharedKeys[elem2] = k2;
    sharedValues[elem2] = v2;
    sharedValid[elem2] = valid2;

    // Sort!
    bitonicSort<Comparator, K, V, IndexType, Power2SortSize>(
      sharedKeys, sharedValues, sharedValid, comp);

    // elem1 and elem2 values might be out-of-range, if the data size we are
    // sorting is smaller than half the power2 size
    if (valid1) {
      keys.data[keyStartOffset + elem1 * keySliceStride] =
        sharedKeys[elem1];
      values.data[valueStartOffset + elem1 * valueSliceStride] =
        sharedValues[elem1];
    }

    if (valid2) {
      keys.data[keyStartOffset + elem2 * keySliceStride] =
        sharedKeys[elem2];
      values.data[valueStartOffset + elem2 * valueSliceStride] =
        sharedValues[elem2];
    }
  }
}

bool should_use_small_sort(const Tensor &self, int64_t dim);
void sortKeyValueInplace(const Tensor& key,
                         const Tensor& value,
                         int dim, bool dir);
}} // at::native
