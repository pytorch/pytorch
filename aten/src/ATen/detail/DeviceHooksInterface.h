#pragma once

#include <ATen/core/Generator.h>
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
namespace at {

struct TORCH_API DeviceHooksInterface {
  virtual ~DeviceHooksInterface() = default;
  virtual const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) {
    C10_THROW_ERROR(NotImplementedError, "must be inherited to implement.");
  }

  virtual at::Device getDeviceFromPtr(void* data) const {
    C10_THROW_ERROR(NotImplementedError, "must be inherited to implement.");
  }
};

struct TORCH_API DeviceHooksArgs {};

TORCH_API void SetDeviceHooksInterface(at::DeviceType t, at::DeviceHooksInterface* hook_);

TORCH_API at::DeviceHooksInterface* GetDeviceHooksInterface(const at::DeviceType& t);

}
