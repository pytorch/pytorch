#include <c10/cuda/impl/CUDAGuardImpl.h>

#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace c10 {
namespace cuda {
namespace impl {

constexpr DeviceType CUDAGuardImpl::static_type;

C10_REGISTER_GUARD_IMPL(CUDA, CUDAGuardImpl);

} // namespace impl
} // namespace cuda
} // namespace c10
