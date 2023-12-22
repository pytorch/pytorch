#pragma once

#include <ATen/detail/XPUHooksInterface.h>

#include <ATen/Generator.h>
#include <c10/util/Optional.h>

namespace at::xpu::detail {

// The real implementation of XPUHooksInterface
struct XPUHooks : public at::XPUHooksInterface {
  XPUHooks(at::XPUHooksArgs) {}
  void initXPU() const override;
  bool hasXPU() const override;
  std::string showConfig() const override;
  int getGlobalIdFromDevice(const at::Device& device) const override;
  Device getDeviceFromPtr(void* data) const override;
  int getNumGPUs() const override;
};

} // namespace at::xpu::detail
