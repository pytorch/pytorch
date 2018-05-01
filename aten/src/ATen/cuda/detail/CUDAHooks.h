#include <ATen/detail/CUDAHooksInterface.h>

namespace at { namespace cuda { namespace detail {

// The real implementation of CUDAHooksInterface
class CUDAHooks : public at::detail::CUDAHooksInterface {

  void doInitCUDA() const override;

};

// Sigh, namespace shenanigans
using at::detail::RegistererCUDAHooksRegistry;
using at::detail::CUDAHooksRegistry;

REGISTER_CUDA_HOOKS(CUDAHooks);

}}} // at::cuda::detail
