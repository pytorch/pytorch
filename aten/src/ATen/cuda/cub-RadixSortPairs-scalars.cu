#include <ATen/cuda/cub-RadixSortPairs.cuh>

namespace at::cuda::cub::detail {

AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, AT_INSTANTIATE_SORT_PAIRS_8)

} // namespace at::cuda::cub::detail
