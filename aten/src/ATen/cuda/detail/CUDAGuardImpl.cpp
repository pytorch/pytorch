#include <ATen/cuda/detail/CUDAGuardImpl.h>

namespace at {
namespace cuda {
namespace detail {

C10_REGISTER_GUARD_IMPL(CUDA, CUDAGuardImpl);

}}} // namespace at::cuda::detail
