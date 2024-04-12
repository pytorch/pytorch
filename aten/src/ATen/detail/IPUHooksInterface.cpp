#include <ATen/detail/IPUHooksInterface.h>

#include <c10/util/CallOnce.h>

namespace at {
namespace detail {

const IPUHooksInterface& getIPUHooks() {
  static std::unique_ptr<IPUHooksInterface> hooks;
  static c10::once_flag once;
  c10::call_once(once, [] {
    hooks = IPUHooksRegistry()->Create("IPUHooks", IPUHooksArgs{});
    if (!hooks) {
      hooks = std::make_unique<IPUHooksInterface>();
    }
  });
  return *hooks;
}

} // namespace detail

C10_DEFINE_REGISTRY(IPUHooksRegistry, IPUHooksInterface, IPUHooksArgs)

} // namespace at
