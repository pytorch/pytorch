#pragma once

#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/detail/AcceleratorHooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/Storage.h>
#include <c10/util/Exception.h>

#include <string>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-parameter")

namespace at {

struct TORCH_API IpcMemHandle {
  std::string handle;
  uint64_t offset;
};

struct TORCH_API PrivateUse1HooksInterface : AcceleratorHooksInterface {
#define FAIL_PRIVATEUSE1HOOKS_FUNC(func)                        \
  TORCH_CHECK_NOT_IMPLEMENTED(                                  \
      false,                                                    \
      "You should register `PrivateUse1HooksInterface`",        \
      "by `RegisterPrivateUse1HooksInterface` and implement `", \
      func,                                                     \
      "` at the same time for PrivateUse1.");

  ~PrivateUse1HooksInterface() override = default;

  bool isBuilt() const override {
    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  bool isAvailable() const override {
    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  const at::Generator& getDefaultGenerator(
      c10::DeviceIndex device_index) const override {
    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  Generator getNewGenerator(
      [[maybe_unused]] DeviceIndex device_index = -1) const override {
    // TODO(FFFrog): Preserved for BC and will be removed in the future.
    if (at::GetGeneratorPrivate().has_value())
      return at::GetGeneratorForPrivateuse1(device_index);

    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  at::Device getDeviceFromPtr(void* data) const override {
    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  bool isPinnedPtr(const void* data) const override {
    return false;
  }

  Allocator* getPinnedMemoryAllocator() const override {
    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  bool hasPrimaryContext(DeviceIndex device_index) const override {
    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  void init() const override {}
  virtual void resizePrivateUse1Bytes(
      const c10::Storage& storage,
      size_t newsize) const {
    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  virtual bool supportsIpc() const {
    return false;
  }

  virtual size_t ipcEventHandleSize() const {
    return 0;
  }

  virtual IpcMemHandle getIpcMemHandle(
      [[maybe_unused]] void* ptr) const {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "getIpcMemHandle is not implemented for this PrivateUse1 device; "
        "implement it in PrivateUse1HooksInterface to enable IPC.");
  }

  virtual std::string getIpcEventHandle() const {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "getIpcEventHandle is not implemented for this PrivateUse1 device; "
        "implement it in PrivateUse1HooksInterface to enable IPC.");
  }

  virtual c10::DataPtr openIpcMemHandle(
      [[maybe_unused]] const std::string& handle) const {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "openIpcMemHandle is not implemented for this PrivateUse1 device; "
        "implement it in PrivateUse1HooksInterface to enable IPC.");
  }

  virtual void waitIpcEvent(
      [[maybe_unused]] const std::string& event_bytes,
      [[maybe_unused]] const c10::Stream& stream) const {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "waitIpcEvent is not implemented for this PrivateUse1 device; "
        "implement it in PrivateUse1HooksInterface to enable IPC.");
  }

#undef FAIL_PRIVATEUSE1HOOKS_FUNC
};

struct TORCH_API PrivateUse1HooksArgs {};

TORCH_API void RegisterPrivateUse1HooksInterface(
    at::PrivateUse1HooksInterface* hook_);

TORCH_API bool isPrivateUse1HooksRegistered();

namespace detail {

TORCH_API const at::PrivateUse1HooksInterface& getPrivateUse1Hooks();

} // namespace detail

} // namespace at

C10_DIAGNOSTIC_POP()
