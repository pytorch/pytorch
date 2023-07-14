#pragma once

#include <ATen/core/Generator.h>
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
namespace at {

struct TORCH_API PrivateUse1HooksInterface {
  virtual ~PrivateUse1HooksInterface() = default;
  virtual const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) {
    C10_THROW_ERROR(NotImplementedError, "must be inherited to implement.");
  }

  virtual at::Device getDeviceFromPtr(void* data) const {
    C10_THROW_ERROR(NotImplementedError, "must be inherited to implement.");
  }
};

struct TORCH_API PrivateUse1HooksArgs {};

TORCH_API void SetPrivateUse1HooksInterface(at::DeviceType t, at::PrivateUse1HooksInterface* hook_);

TORCH_API at::PrivateUse1HooksInterface* GetPrivateUse1HooksInterface(const at::DeviceType& t);

}
