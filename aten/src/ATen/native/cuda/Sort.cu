#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/Sort.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/cuda/SortDecl.cuh>
#include <ATen/Dispatch_v2.h>

namespace at::native {

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
  #ifdef USE_ROCM
      if (at::cuda::warp_size() == 32) {
        sortCommon<idx_unsigned_t>(WarpMergeSort<128, 32>{}, key, value, dim, descending);
      }
      else {
        sortCommon<idx_unsigned_t>(WarpMergeSort<128, 64>{}, key, value, dim, descending);
      }
  #else
      sortCommon<idx_unsigned_t>(WarpMergeSort<128, C10_WARP_SIZE>{}, key, value, dim, descending);
  #endif
  #endif
    } else {
      sortCommon<idx_unsigned_t>(MediumRadixSort{}, key, value, dim, descending);
    }
  }), kChar, kByte, kShort, kUInt16, kInt, kLong);

}

}  // namespace at::native
