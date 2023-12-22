#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <ATen/core/Generator.h>
#include <c10/util/Registry.h>

#include <cstddef>
#include <functional>
#include <memory>

namespace at {

struct TORCH_API XPUHooksInterface {
  virtual ~XPUHooksInterface() {}

  virtual void initXPU() const {
    TORCH_CHECK(false, "Cannot initialize XPU without ATen_xpu library.");
  }

  virtual bool hasXPU() const {
    return false;
  }

  virtual std::string showConfig() const {
    TORCH_CHECK(false, "Cannot query detailed XPU version without ATen_xpu library.");
  }

  virtual int getGlobalIdFromDevice(const Device& device) const {
    TORCH_CHECK(false, "Cannot get XPU global device id without ATen_xpu library.");
  }

  virtual Generator getXPUGenerator(C10_UNUSED DeviceIndex device_index = -1) const {
    TORCH_CHECK(false, "Cannot get XPU generator without ATen_xpu library.");
  }

  virtual const Generator& getDefaultXPUGenerator(C10_UNUSED DeviceIndex device_index = -1) const {
    TORCH_CHECK(false, "Cannot get default XPU generator without ATen_xpu library.");
  }

  virtual int getNumGPUs() const {
    return 0;
  }

  virtual Device getDeviceFromPtr(void* /*data*/) const {
    TORCH_CHECK(false, "Cannot get device of pointer on XPU without ATen_xpu library.");
  }
};

struct TORCH_API XPUHooksArgs {};

C10_DECLARE_REGISTRY(XPUHooksRegistry, XPUHooksInterface, XPUHooksArgs);
#define REGISTER_XPU_HOOKS(clsname) \
  C10_REGISTER_CLASS(XPUHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const XPUHooksInterface& getXPUHooks();
} // namespace detail
} // namespace at
