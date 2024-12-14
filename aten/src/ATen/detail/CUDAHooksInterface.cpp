#include "c10/util/Registry.h"
#include <ATen/detail/CUDAHooksInterface.h>

#include <memory>

namespace at {
namespace detail {

// NB: We purposely leak the CUDA hooks object.  This is because under some
// situations, we may need to reference the CUDA hooks while running destructors
// of objects which were constructed *prior* to the first invocation of
// getCUDAHooks.  The example which precipitated this change was the fused
// kernel cache in the JIT.  The kernel cache is a global variable which caches
// both CPU and CUDA kernels; CUDA kernels must interact with CUDA hooks on
// destruction.  Because the kernel cache handles CPU kernels too, it can be
// constructed before we initialize CUDA; if it contains CUDA kernels at program
// destruction time, you will destruct the CUDA kernels after CUDA hooks has
// been unloaded.  In principle, we could have also fixed the kernel cache store
// CUDA kernels in a separate global variable, but this solution is much
// simpler.
//
// CUDAHooks doesn't actually contain any data, so leaking it is very benign;
// you're probably losing only a word (the vptr in the allocated object.)

const CUDAHooksInterface& getCUDAHooks() {
  auto create_impl = [] {
#if !defined C10_MOBILE
    auto hooks = CUDAHooksRegistry()->Create("CUDAHooks", CUDAHooksArgs{});
    if (hooks) {
      return hooks;
    }
#endif
    return std::make_unique<CUDAHooksInterface>();
  };
  // NB: The static initialization here implies that if you try to call any CUDA
  // functionality before libATen_cuda.so is loaded, CUDA is permanently
  // disabled for that copy of ATen.  In principle, we can relax this
  // restriction, but you might have to fix some code.  See getVariableHooks()
  // for an example where we relax this restriction (but if you try to avoid
  // needing a lock, be careful; it doesn't look like Registry.h is thread
  // safe...)
  static auto hooks = create_impl();
  return *hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(CUDAHooksRegistry, CUDAHooksInterface, CUDAHooksArgs)

} // namespace at
