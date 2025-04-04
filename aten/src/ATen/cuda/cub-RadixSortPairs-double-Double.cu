#include <ATen/cuda/cub-RadixSortPairs.cuh>

namespace at::cuda::cub::detail {

AT_INSTANTIATE_SORT_PAIRS_8(double, Double)

} // namespace at::cuda::cub::detail
