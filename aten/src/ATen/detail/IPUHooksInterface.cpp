#include <ATen/detail/IPUHooksInterface.h>

#include <c10/util/CallOnce.h>

namespace at {
namespace detail {

const IPUHooksInterface& getIPUHooks() {
  auto create_impl = [] {
    auto hooks = IPUHooksRegistry()->Create("IPUHooks", IPUHooksArgs{});
    if (hooks) {
      return hooks;
    }
    return std::make_unique<IPUHooksInterface>();
  };
  static auto hooks = create_impl();
  return *hooks;
}

} // namespace detail

C10_DEFINE_REGISTRY(IPUHooksRegistry, IPUHooksInterface, IPUHooksArgs)

} // namespace at
