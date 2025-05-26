#include <ATen/detail/MAIAHooksInterface.h>

namespace at {
namespace detail {

// See getCUDAHooks for some more commentary
const MAIAHooksInterface& getMAIAHooks() {
  auto create_impl = [] {
    auto hooks = MAIAHooksRegistry()->Create("MAIAHooks", {});
    if (hooks) {
      return hooks;
    }
    return std::make_unique<MAIAHooksInterface>();
  };
  static auto hooks = create_impl();
  return *hooks;
}
} // namespace detail

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(MAIAHooksRegistry, MAIAHooksInterface, MAIAHooksArgs)

} // namespace at
