#include <ATen/detail/HPUHooksInterface.h>
#include <c10/util/CallOnce.h>

namespace at {
namespace detail {

static HPUHooksInterface* hpu_hooks = nullptr;

TORCH_API const at::HPUHooksInterface& getHPUHooks() {
  static c10::once_flag once;
  c10::call_once(once, [] {
    hpu_hooks =
        HPUHooksRegistry()->Create("HPUHooks", HPUHooksArgs{}).release();
    TORCH_CHECK(
        hpu_hooks != nullptr,
        "Missing registration for HPUHooks in HPUHooksRegistry.")
  });
  return *hpu_hooks;
}

} // namespace detail

C10_DEFINE_REGISTRY(HPUHooksRegistry, HPUHooksInterface, HPUHooksArgs)

} // namespace at
