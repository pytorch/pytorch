#include <ATen/detail/MTIAHooksInterface.h>

#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {

static MTIAHooksInterface* MTIA_hooks = nullptr;

const MTIAHooksInterface &getMTIAHooks() {
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
