#include <ATen/detail/CUDAHooksInterface.h>

namespace at { namespace detail {

const CUDAHooksInterface& getCUDAHooks() {
  static std::unique_ptr<CUDAHooksInterface> cuda_hooks;
  if (!cuda_hooks) {
    cuda_hooks = CUDAHooksRegistry()->Create("CUDAHooks");
    if (!cuda_hooks) {
      cuda_hooks = std::unique_ptr<CUDAHooksInterface>(new CUDAHooksInterface());
    }
  }
  return *cuda_hooks;
}

AT_DEFINE_REGISTRY(CUDAHooksRegistry, CUDAHooksInterface);

}} // namespace at::detail
