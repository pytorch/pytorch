#include <ATen/detail/HIPHooksInterface.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {

// See getCUDAHooks for some more commentary
const HIPHooksInterface& getHIPHooks() {
  static std::unique_ptr<HIPHooksInterface> hip_hooks;
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
