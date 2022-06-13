#include <ATen/detail/ORTHooksInterface.h>

#include <c10/util/Exception.h>
#include <c10/util/CallOnce.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {

// See getCUDAHooks for some more commentary
const ORTHooksInterface& getORTHooks() {
  static std::unique_ptr<ORTHooksInterface> ort_hooks;
  static c10::once_flag once;
  c10::call_once(once, [] {
    ort_hooks = ORTHooksRegistry()->Create("ORTHooks", {});
    if (!ort_hooks) {
      ort_hooks =
          // NOLINTNEXTLINE(modernize-make-unique)
          std::unique_ptr<ORTHooksInterface>(new ORTHooksInterface());
    }
  });
  return *ort_hooks;
}
} // namespace detail

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(ORTHooksRegistry, ORTHooksInterface, ORTHooksArgs)

} // namespace at
