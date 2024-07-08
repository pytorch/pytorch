#include <ATen/detail/MTIAHooksInterface.h>

#include <c10/util/CallOnce.h>

#include <memory>

namespace at {
namespace detail {

const MTIAHooksInterface& getMTIAHooks() {
  static std::unique_ptr<MTIAHooksInterface> mtia_hooks = nullptr;
  static c10::once_flag once;
  c10::call_once(once, [] {
    mtia_hooks = MTIAHooksRegistry()->Create("MTIAHooks", MTIAHooksArgs{});
    if (!mtia_hooks) {
      mtia_hooks = std::make_unique<MTIAHooksInterface>();
    }
  });
  return *mtia_hooks;
}

bool isMTIAHooksBuilt() {
  return MTIAHooksRegistry()->Has("MTIAHooks");
}

} // namespace detail

C10_DEFINE_REGISTRY(MTIAHooksRegistry, MTIAHooksInterface, MTIAHooksArgs)

} // namespace at
