#include <ATen/detail/MAIAHooksInterface.h>

#include <c10/util/CallOnce.h>
#include <c10/util/Registry.h>

#include <cstddef>
#include <memory>

namespace at {
namespace detail {

// See getCUDAHooks for some more commentary
const MAIAHooksInterface& getMAIAHooks() {
  static std::unique_ptr<MAIAHooksInterface> maia_hooks;
  static c10::once_flag once;
  c10::call_once(once, [] {
    maia_hooks = MAIAHooksRegistry()->Create("MAIAHooks", {});
    if (!maia_hooks) {
      maia_hooks = std::make_unique<MAIAHooksInterface>();
    }
  });
  return *maia_hooks;
}
} // namespace detail

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(MAIAHooksRegistry, MAIAHooksInterface, MAIAHooksArgs)

} // namespace at
