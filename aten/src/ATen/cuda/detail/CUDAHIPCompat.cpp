#include <ATen/cuda/detail/CUDAHIPCompat.h>

namespace at { namespace cuda { namespace detail {

#if AT_ROCM_ENABLED()

// THIS IS A MASSIVE HACK
C10_REGISTER_GUARD_IMPL(CUDA, HIPGuardImplMasqueradingAsCUDA);

#endif

}}}
