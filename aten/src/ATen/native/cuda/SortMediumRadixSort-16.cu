#include <ATen/native/cuda/SortImpl.cuh>

namespace at::native {
  template void sortCommon<uint16_t>(MediumRadixSort sorter, const TensorBase &key, const TensorBase &value,
                int dim, bool descending);
}
