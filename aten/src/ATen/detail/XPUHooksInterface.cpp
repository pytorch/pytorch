#include <ATen/detail/XPUHooksInterface.h>

#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {

static XPUHooksInterface *xpu_hooks = nullptr;

const XPUHooksInterface &getXPUHooks() {
  static c10::once_flag once;
  c10::call_once(once, [] {
    xpu_hooks =
        XPUHooksRegistry()->Create("XPUHooks", XPUHooksArgs{}).release();
    if (!xpu_hooks) {
      xpu_hooks = new XPUHooksInterface();
    }
  });
  return *xpu_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(XPUHooksRegistry, XPUHooksInterface, XPUHooksArgs)

} // namespace at
