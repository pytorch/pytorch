#include <ATen/cuda/cub-RadixSortPairs.cuh>

namespace at::cuda::cub::detail {

AT_INSTANTIATE_SORT_PAIRS_8(decltype(::c10::impl::ScalarTypeToCPPType<at::ScalarType::Half>::t), at::ScalarType::Half)

} // namespace at::cuda::cub::detail
