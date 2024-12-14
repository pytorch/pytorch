#include <ATen/detail/HIPHooksInterface.h>

#include <c10/util/Registry.h>

#include <memory>

namespace at {
namespace detail {

// See getCUDAHooks for some more commentary
const HIPHooksInterface& getHIPHooks() {
  auto create_impl = [] {
#if !defined C10_MOBILE
    auto hooks = HIPHooksRegistry()->Create("HIPHooks", HIPHooksArgs{});
    if (hooks) {
      return hooks;
    }
#endif
    return std::make_unique<HIPHooksInterface>();
  };
  static auto hooks = create_impl();
  return *hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(HIPHooksRegistry, HIPHooksInterface, HIPHooksArgs)

} // namespace at
