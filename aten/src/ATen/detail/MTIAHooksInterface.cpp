#include <ATen/detail/MTIAHooksInterface.h>

#include <c10/util/CallOnce.h>

#include <cstddef>
#include <memory>

namespace at {
namespace detail {


const MTIAHooksInterface &getMTIAHooks() {
  static MTIAHooksInterface* MTIA_hooks = nullptr;
  static c10::once_flag once;
  c10::call_once(once, [] {
    MTIA_hooks =
        MTIAHooksRegistry()->Create("MTIAHooks", MTIAHooksArgs{}).release();
    if (!MTIA_hooks) {
      MTIA_hooks = new MTIAHooksInterface();
    }
  });
  return *MTIA_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(MTIAHooksRegistry, MTIAHooksInterface, MTIAHooksArgs)

} // namespace at
