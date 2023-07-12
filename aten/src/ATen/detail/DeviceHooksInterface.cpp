#include <ATen/detail/DeviceHooksInterface.h>

namespace at {

TORCH_API DeviceHooksInterface* device_hooks[at::COMPILE_TIME_MAX_DEVICE_TYPES];
// TODO: rafactor device hooks interface for other device, like cuda/mps.
TORCH_API void SetDeviceHooksInterface(at::DeviceType t, at::DeviceHooksInterface* hook_) {
  TORCH_CHECK(t == at::DeviceType::PrivateUse1, "SetDeviceHooksInterface only support `PrivateUse1` now.");
  device_hooks[static_cast<int>(t)] = hook_;
}

TORCH_API at::DeviceHooksInterface* GetDeviceHooksInterface(const at::DeviceType& t) {
  auto* hook = device_hooks[static_cast<int>(t)];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(hook, "DeviceHooksInterface for ", t, " is not set.");
  return hook;
}

}
