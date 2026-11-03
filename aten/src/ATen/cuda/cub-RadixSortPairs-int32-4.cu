#include <ATen/cuda/cub-RadixSortPairs.cuh>

namespace at::cuda::cub::detail {

AT_INSTANTIATE_SORT_PAIRS(int32_t, 4)

#if defined(USE_ROCM)
AT_INSTANTIATE_SORT_PAIRS_ONESWEEP(int32_t, 4)
#endif

} // namespace at::cuda::cub::detail
