#include <ATen/cuda/cub-RadixSortPairs.cuh>

namespace at::cuda::cub::detail {

AT_INSTANTIATE_SORT_PAIRS(c10::BFloat16, 8)
AT_INSTANTIATE_SORT_PAIRS(c10::BFloat16, 4)
AT_INSTANTIATE_SORT_PAIRS(c10::BFloat16, 2)
AT_INSTANTIATE_SORT_PAIRS(c10::BFloat16, 1)


} // namespace at::cuda::cub::detail
