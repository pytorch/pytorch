#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/Sort.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/cuda/SortImpl.cuh>
#include <type_traits>

namespace at::native {

extern template void sortCommon<uint8_t>(SmallBitonicSort sorter, const TensorBase &key, const TensorBase &value, int dim, bool descending);
extern template void sortCommon<uint16_t>(SmallBitonicSort sorter, const TensorBase &key, const TensorBase &value, int dim, bool descending);
extern template void sortCommon<uint32_t>(SmallBitonicSort sorter, const TensorBase &key, const TensorBase &value, int dim, bool descending);
extern template void sortCommon<uint64_t>(SmallBitonicSort sorter, const TensorBase &key, const TensorBase &value, int dim, bool descending);

#if HAS_WARP_MERGE_SORT()
extern template void sortCommon<uint8_t>(WarpMergeSort<128> sorter, const TensorBase &key, const TensorBase &value, int dim, bool descending);
extern template void sortCommon<uint16_t>(WarpMergeSort<128> sorter, const TensorBase &key, const TensorBase &value, int dim, bool descending);
extern template void sortCommon<uint32_t>(WarpMergeSort<128> sorter, const TensorBase &key, const TensorBase &value, int dim, bool descending);
extern template void sortCommon<uint64_t>(WarpMergeSort<128> sorter, const TensorBase &key, const TensorBase &value, int dim, bool descending);
#endif

extern template void sortCommon<uint8_t>(MediumRadixSort sorter, const TensorBase &key, const TensorBase &value, int dim, bool descending);
extern template void sortCommon<uint16_t>(MediumRadixSort sorter, const TensorBase &key, const TensorBase &value, int dim, bool descending);
extern template void sortCommon<uint32_t>(MediumRadixSort sorter, const TensorBase &key, const TensorBase &value, int dim, bool descending);
extern template void sortCommon<uint64_t>(MediumRadixSort sorter, const TensorBase &key, const TensorBase &value, int dim, bool descending);

void sortKeyValueInplace(
    const TensorBase& key,
    const TensorBase& value,
    int64_t dim,
    bool descending,
    bool stable) {
  const auto sort_size = key.size(dim);
  AT_DISPATCH_V2(value.scalar_type(), "sortKeyValueInplace", AT_WRAP([&] {
    using idx_unsigned_t = std::make_unsigned_t<scalar_t>;

    if (sort_size <= 1) {
      return; // Already sorted
    } else if (!stable && sort_size <= 32) {
      // NOTE: Bitonic sort is unstable
      sortCommon<idx_unsigned_t>(SmallBitonicSort{}, key, value, dim, descending);
  #if HAS_WARP_MERGE_SORT()
    } else if (sort_size <= 128) {
      sortCommon<idx_unsigned_t>(WarpMergeSort<128>{}, key, value_viewed, dim, descending);
  #endif
    } else {
      sortCommon<idx_unsigned_t>(MediumRadixSort{}, key, value, dim, descending);
    }
  }), kChar, kByte, kShort, kUInt16, kInt, kLong);

}

}  // namespace at::native
