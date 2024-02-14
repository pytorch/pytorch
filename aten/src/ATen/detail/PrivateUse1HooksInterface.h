#pragma once

#include <ATen/core/Generator.h>
#include <ATen/detail/AcceleratorHooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/Storage.h>
#include <c10/util/Exception.h>
namespace at {

struct TORCH_API PrivateUse1HooksInterface : AcceleratorHooksInterface {
  virtual ~PrivateUse1HooksInterface() override = default;
  virtual const at::Generator& getDefaultGenerator(
      c10::DeviceIndex device_index) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `getDefaultGenerator`.");
  }

  virtual at::Device getDeviceFromPtr(void* data) const {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `getDeviceFromPtr`.");
  }

  virtual Allocator* getPinnedMemoryAllocator() const {
    TORCH_CHECK(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `getPinnedMemoryAllocator`.");
  }

  virtual bool hasPrimaryContext(DeviceIndex device_index) const override {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `hasPrimaryContext`.");
  }

  virtual void initPrivateUse1() const {}
  virtual void resizePrivateUse1Bytes(const c10::Storage &storage, size_t newsize) const {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `resizePrivateUse1Bytes`.");
  }
};

struct TORCH_API PrivateUse1HooksArgs {};

TORCH_API void RegisterPrivateUse1HooksInterface(
    at::PrivateUse1HooksInterface* hook_);

TORCH_API at::PrivateUse1HooksInterface* GetPrivateUse1HooksInterface();

TORCH_API bool isPrivateUse1HooksRegistered();

namespace detail {

TORCH_API const at::PrivateUse1HooksInterface& getPrivateUse1Hooks();

} // namespace detail

} // namespace at
