#include <ATen/detail/XPUHooksInterface.h>

#include <c10/util/CallOnce.h>

namespace at {
namespace detail {

const XPUHooksInterface& getXPUHooks() {
  auto create_impl = [] {
    auto hooks = XPUHooksRegistry()->Create("XPUHooks", XPUHooksArgs{});
    if (hooks) {
      return hooks;
    }
    return std::make_unique<XPUHooksInterface>();
  };
  static auto hooks = create_impl();
  return *hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(XPUHooksRegistry, XPUHooksInterface, XPUHooksArgs)

} // namespace at
