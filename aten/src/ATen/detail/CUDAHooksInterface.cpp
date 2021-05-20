#include <ATen/detail/CUDAHooksInterface.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <memory>
#include <mutex>

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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static CUDAHooksInterface* cuda_hooks = nullptr;

const CUDAHooksInterface& getCUDAHooks() {
  // NB: The once_flag here implies that if you try to call any CUDA
  // functionality before libATen_cuda.so is loaded, CUDA is permanently
  // disabled for that copy of ATen.  In principle, we can relax this
  // restriction, but you might have to fix some code.  See getVariableHooks()
  // for an example where we relax this restriction (but if you try to avoid
  // needing a lock, be careful; it doesn't look like Registry.h is thread
  // safe...)
  static std::once_flag once;
  std::call_once(once, [] {
    cuda_hooks = CUDAHooksRegistry()->Create("CUDAHooks", CUDAHooksArgs{}).release();
    if (!cuda_hooks) {
      cuda_hooks = new CUDAHooksInterface();
    }
  });
  return *cuda_hooks;
}
} // namespace detail

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(CUDAHooksRegistry, CUDAHooksInterface, CUDAHooksArgs)

} // namespace at
