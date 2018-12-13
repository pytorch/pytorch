#include <c10/cuda/impl/CUDAGuardImpl.h>

namespace c10 {
namespace cuda {
namespace impl {

constexpr DeviceType CUDAGuardImpl::static_type;

C10_REGISTER_GUARD_IMPL(CUDA, CUDAGuardImpl);

}}} // namespace c10::cuda::detail
