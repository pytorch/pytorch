#include <ATen/native/cuda/SortImpl.cuh>

namespace at::native {
#if HAS_WARP_MERGE_SORT()
  template void sortCommon<uint64_t>(WarpMergeSort<128> sorter, const TensorBase &key, const TensorBase &value,
                int dim, bool descending);
#endif
}
