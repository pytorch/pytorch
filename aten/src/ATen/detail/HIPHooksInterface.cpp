#include <ATen/detail/HIPHooksInterface.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {

const HIPHooksInterface& getHIPHooks() {
  static std::unique_ptr<HIPHooksInterface> hip_hooks;
  // NB: The once_flag here implies that if you try to call any HIP
  // functionality before libATen_cuda.so is loaded, HIP is permanently
  // disabled for that copy of ATen.  In principle, we can relax this
  // restriction, but you might have to fix some code.  See getVariableHooks()
  // for an example where we relax this restriction (but if you try to avoid
  // needing a lock, be careful; it doesn't look like Registry.h is thread
  // safe...)
  static std::once_flag once;
  std::call_once(once, [] {
    hip_hooks = HIPHooksRegistry()->Create("HIPHooks", HIPHooksArgs{});
    if (!hip_hooks) {
      hip_hooks =
          std::unique_ptr<HIPHooksInterface>(new HIPHooksInterface());
    }
  });
  return *hip_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(HIPHooksRegistry, HIPHooksInterface, HIPHooksArgs)

} // namespace at
