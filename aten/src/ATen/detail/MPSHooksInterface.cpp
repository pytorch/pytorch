//  Copyright © 2022 Apple Inc.

#include <ATen/detail/MPSHooksInterface.h>

namespace at {
namespace detail {

const MPSHooksInterface& getMPSHooks() {
  auto create_impl = [] {
#if !defined C10_MOBILE
    auto hooks = MPSHooksRegistry()->Create("MPSHooks", MPSHooksArgs{});
    if (hooks) {
      return hooks;
    }
#endif
    return std::make_unique<MPSHooksInterface>();
  };
  static auto hooks = create_impl();
  return *hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(MPSHooksRegistry, MPSHooksInterface, MPSHooksArgs)

} // namespace at
