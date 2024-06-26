//  Copyright © 2022 Apple Inc.

#include <ATen/detail/MPSHooksInterface.h>
#include <c10/util/CallOnce.h>

namespace at {
namespace detail {

const MPSHooksInterface& getMPSHooks() {
  static std::unique_ptr<MPSHooksInterface> mps_hooks;
#if !defined C10_MOBILE
  static c10::once_flag once;
  c10::call_once(once, [] {
    mps_hooks = MPSHooksRegistry()->Create("MPSHooks", MPSHooksArgs{});
    if (!mps_hooks) {
      mps_hooks = std::make_unique<MPSHooksInterface>();
    }
  });
#else
  if (mps_hooks == nullptr) {
    mps_hooks = std::make_unique<MPSHooksInterface>();
  }
#endif
  return *mps_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(MPSHooksRegistry, MPSHooksInterface, MPSHooksArgs)

} // namespace at
