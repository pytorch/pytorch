#include <ATen/cuda/cub-RadixSortPairs.cuh>

namespace at::cuda::cub::detail {

AT_INSTANTIATE_SORT_PAIRS_8(uint8_t, Byte)

} // namespace at::cuda::cub::detail
