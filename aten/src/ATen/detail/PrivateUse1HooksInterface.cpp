#include <ATen/detail/PrivateUse1HooksInterface.h>

namespace at {

static PrivateUse1HooksInterface* privateuse1_hooks = nullptr;
static std::mutex _hooks_mutex_lock;

TORCH_API void SetPrivateUse1HooksInterface(at::DeviceType t, at::PrivateUse1HooksInterface* hook_) {
  std::lock_guard<std::mutex> lock(_hooks_mutex_lock);
  TORCH_CHECK(privateuse1_hooks == nullptr, "PrivateUse1HooksInterface only could be called once.")
  TORCH_CHECK(t == at::DeviceType::PrivateUse1, "SetPrivateUse1HooksInterface only support `PrivateUse1`.");
  privateuse1_hooks = hook_;
}

TORCH_API at::PrivateUse1HooksInterface* GetPrivateUse1HooksInterface(const at::DeviceType& t) {
  TORCH_CHECK(t == at::DeviceType::PrivateUse1, "GetPrivateUse1HooksInterface only support `PrivateUse1`.");
  TORCH_CHECK(privateuse1_hooks != nullptr, "Please SetPrivateUse1HooksInterface first.")
  return privateuse1_hooks;
}

}
