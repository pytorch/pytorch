#include <ATen/cuda/cub-RadixSortPairs.cuh>

namespace at::cuda::cub::detail {

AT_INSTANTIATE_SORT_PAIRS_8(int8_t, Char)

} // namespace at::cuda::cub::detail
