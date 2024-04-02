#pragma once

#include <ATen/core/Generator.h>
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
namespace at {

struct TORCH_API PrivateUse1HooksInterface {
  virtual ~PrivateUse1HooksInterface() = default;
  virtual const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `getDefaultGenerator`.");
  }

  virtual at::Device getDeviceFromPtr(void* data) const {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `getDeviceFromPtr`.");
  }
};

struct TORCH_API PrivateUse1HooksArgs {};

TORCH_API void RegisterPrivateUse1HooksInterface(at::PrivateUse1HooksInterface* hook_);

TORCH_API at::PrivateUse1HooksInterface* GetPrivateUse1HooksInterface();

}
