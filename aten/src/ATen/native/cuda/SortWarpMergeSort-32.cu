#include <ATen/native/cuda/SortImpl.cuh>

namespace at::native {
#if HAS_WARP_MERGE_SORT()
#ifdef USE_ROCM
  template void sortCommon<uint32_t>(WarpMergeSort<128, 32> sorter, const TensorBase &key, const TensorBase &value,
                int dim, bool descending);
  template void sortCommon<uint32_t>(WarpMergeSort<128, 64> sorter, const TensorBase &key, const TensorBase &value,
                int dim, bool descending);
#else
  template void sortCommon<uint32_t>(WarpMergeSort<128, C10_WARP_SIZE> sorter, const TensorBase &key, const TensorBase &value,
                int dim, bool descending);
#endif
#endif
}
