#include <ATen/cuda/cub-RadixSortPairs.cuh>

namespace at::cuda::cub::detail {

AT_INSTANTIATE_SORT_PAIRS(c10::BFloat16, 8)

} // namespace at::cuda::cub::detail
