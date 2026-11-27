#include <ATen/detail/HPUHooksInterface.h>

namespace at {
namespace detail {

TORCH_API const at::HPUHooksInterface& getHPUHooks() {
  auto create_impl = [] {
    auto hooks = HPUHooksRegistry()->Create("HPUHooks", HPUHooksArgs{});
    if (hooks) {
      return hooks;
    }
    return std::make_unique<HPUHooksInterface>();
  };
  static auto hooks = create_impl();
  return *hooks;
}

bool isHPUHooksBuilt() {
  return HPUHooksRegistry()->Has("HPUHooks");
}

} // namespace detail

C10_DEFINE_REGISTRY(HPUHooksRegistry, HPUHooksInterface, HPUHooksArgs)

} // namespace at
