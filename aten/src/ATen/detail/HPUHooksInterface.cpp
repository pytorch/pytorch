#include <ATen/detail/HPUHooksInterface.h>
#include <c10/util/CallOnce.h>
#include <memory>

namespace at {
namespace detail {

TORCH_API const at::HPUHooksInterface& getHPUHooks() {
  static std::unique_ptr<HPUHooksInterface> hpu_hooks;
  static c10::once_flag once;
  c10::call_once(once, [] {
    hpu_hooks = HPUHooksRegistry()->Create("HPUHooks", HPUHooksArgs{});
    if (!hpu_hooks) {
      hpu_hooks = std::make_unique<HPUHooksInterface>();
    }
  });
  return *hpu_hooks;
}

} // namespace detail

C10_DEFINE_REGISTRY(HPUHooksRegistry, HPUHooksInterface, HPUHooksArgs)

} // namespace at
