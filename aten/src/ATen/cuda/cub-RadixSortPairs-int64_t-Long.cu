#include <ATen/cuda/cub-RadixSortPairs.cuh>

namespace at::cuda::cub::detail {

AT_INSTANTIATE_SORT_PAIRS_8(int64_t, at::ScalarType::Long)

} // namespace at::cuda::cub::detail
