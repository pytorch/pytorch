#include <ATen/native/cuda/SortImpl.cuh>

namespace at::native {
  template void sortCommon<uint64_t>(SmallBitonicSort sorter, const TensorBase &key, const TensorBase &value,
                int dim, bool descending);
}
