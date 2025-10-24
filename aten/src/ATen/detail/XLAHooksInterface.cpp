#include <ATen/detail/XLAHooksInterface.h>

namespace at {
namespace detail {

const XLAHooksInterface& getXLAHooks() {
  auto create_impl = [] {
    // Create XLA hooks using the registry
    auto hooks = XLAHooksRegistry()->Create("torch_xla::detail::XLAHooks", XLAHooksArgs{});
    if (hooks) {
      return hooks;
    }
    // If hooks creation fails, fall back to default implementation
    return std::make_unique<XLAHooksInterface>();
  };
  static auto hooks = create_impl();
  return *hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(XLAHooksRegistry, XLAHooksInterface, XLAHooksArgs)

} // namespace at
