#include <ATen/detail/MTIAHooksInterface.h>

namespace at {
namespace detail {

const MTIAHooksInterface& getMTIAHooks() {
  auto create_impl = [] {
    auto hooks = MTIAHooksRegistry()->Create("MTIAHooks", MTIAHooksArgs{});
    if (hooks) {
      return hooks;
    }
    return std::make_unique<MTIAHooksInterface>();
  };
  static auto hooks = create_impl();
  return *hooks;
}

bool isMTIAHooksBuilt() {
  return MTIAHooksRegistry()->Has("MTIAHooks");
}

} // namespace detail

bool MTIAHooksInterface::isAvailable() const {
  return detail::isMTIAHooksBuilt() && detail::getMTIAHooks().deviceCount() > 0;
}

C10_DEFINE_REGISTRY(MTIAHooksRegistry, MTIAHooksInterface, MTIAHooksArgs)

} // namespace at
