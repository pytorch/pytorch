#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/FakeGuardImpl.h>
#include <c10/core/impl/alloc_cpu.h>
#include <ATen/OpaqueTensorImpl.h>

namespace at {

static PrivateUse1HooksInterface* privateuse1_hooks = nullptr;
static std::mutex _hooks_mutex_lock;

TORCH_API void RegisterPrivateUse1HooksInterface(at::PrivateUse1HooksInterface* hook_) {
  std::lock_guard<std::mutex> lock(_hooks_mutex_lock);
  TORCH_CHECK(privateuse1_hooks == nullptr, "PrivateUse1HooksInterface only could be registered once.");
  privateuse1_hooks = hook_;
}

TORCH_API bool isPrivateUse1HooksRegistered() {
  return privateuse1_hooks != nullptr;
}

namespace detail {

TORCH_API const at::PrivateUse1HooksInterface& getPrivateUse1Hooks() {
  TORCH_CHECK(
      privateuse1_hooks != nullptr,
      "Please register PrivateUse1HooksInterface by `RegisterPrivateUse1HooksInterface` first.");
  return *privateuse1_hooks;
}

} // namespace detail

struct OpenRegHooksInterface : public at::PrivateUse1HooksInterface {
  bool hasPrimaryContext(c10::DeviceIndex device_index) const override { return true; }
};

void setupPrivateUse1ForPythonUse() {
  static OpenRegHooksInterface interface;
  if (privateuse1_hooks == nullptr) {
    at::RegisterPrivateUse1HooksInterface(&interface);
  }
  static ::c10::impl::DeviceGuardImplRegistrar 
      g_PrivateUse1(c10::DeviceType::PrivateUse1, new c10::impl::NoOpDeviceGuardImpl<c10::DeviceType::PrivateUse1>(
        /* default_index */ 0, 
        /* fail_on_event_functions */ false));
}

} // namespace at

