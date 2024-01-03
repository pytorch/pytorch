#include <c10/cuda/impl/CUDAGuardImpl.h>

namespace c10 {
namespace cuda {
namespace impl {

C10_REGISTER_GUARD_IMPL(CUDA, CUDAGuardImpl);

} // namespace impl
} // namespace cuda
} // namespace c10
