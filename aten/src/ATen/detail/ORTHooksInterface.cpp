#include <ATen/detail/ORTHooksInterface.h>

#include <c10/util/CallOnce.h>
#include <c10/util/Registry.h>

#include <cstddef>
#include <memory>

namespace at {
namespace detail {

// See getCUDAHooks for some more commentary
const ORTHooksInterface& getORTHooks() {
  static std::unique_ptr<ORTHooksInterface> ort_hooks;
  static c10::once_flag once;
  c10::call_once(once, [] {
    ort_hooks = ORTHooksRegistry()->Create("ORTHooks", {});
    if (!ort_hooks) {
      ort_hooks = std::make_unique<ORTHooksInterface>();
    }
  });
  return *ort_hooks;
}
} // namespace detail

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(ORTHooksRegistry, ORTHooksInterface, ORTHooksArgs)

} // namespace at
