#include <ATen/cuda/cub-RadixSortPairs.cuh>

namespace at::cuda::cub::detail {

AT_INSTANTIATE_SORT_PAIRS(uint16_t, 8)
AT_INSTANTIATE_SORT_PAIRS(uint16_t, 4)
AT_INSTANTIATE_SORT_PAIRS(uint16_t, 2)
AT_INSTANTIATE_SORT_PAIRS(uint16_t, 1)


} // namespace at::cuda::cub::detail
