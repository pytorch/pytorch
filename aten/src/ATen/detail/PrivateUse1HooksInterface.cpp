#include <ATen/detail/PrivateUse1HooksInterface.h>

namespace at {

static PrivateUse1HooksInterface* privateuse1_hooks = nullptr;
static std::mutex _hooks_mutex_lock;

TORCH_API void RegisterPrivateUse1HooksInterface(at::PrivateUse1HooksInterface* hook_) {
  std::lock_guard<std::mutex> lock(_hooks_mutex_lock);
  TORCH_CHECK(privateuse1_hooks == nullptr, "PrivateUse1HooksInterface only could be registered once.");
  privateuse1_hooks = hook_;
}

TORCH_API at::PrivateUse1HooksInterface* GetPrivateUse1HooksInterface() {
  TORCH_CHECK(
      privateuse1_hooks != nullptr,
      "Please register PrivateUse1HooksInterface by `RegisterPrivateUse1HooksInterface` first.");
  return privateuse1_hooks;
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

} // namespace at
