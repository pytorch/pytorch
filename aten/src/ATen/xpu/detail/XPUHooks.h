#pragma once

#include <ATen/detail/XPUHooksInterface.h>

namespace at::xpu::detail {

// The real implementation of XPUHooksInterface
struct XPUHooks : public at::XPUHooksInterface {
  XPUHooks(at::XPUHooksArgs) {}
  void init() const override;
  bool hasXPU() const override;
  std::string showConfig() const override;
  int32_t getGlobalIdxFromDevice(const at::Device& device) const override;
  const Generator& getDefaultGenerator(
      DeviceIndex device_index = -1) const override;
  Generator getNewGenerator(DeviceIndex device_index = -1) const override;
  Device getDeviceFromPtr(void* data) const override;
  c10::DeviceIndex getNumGPUs() const override;
  DeviceIndex current_device() const override;
  void deviceSynchronize(DeviceIndex device_index) const override;
  Allocator* getPinnedMemoryAllocator() const override;
  bool isPinnedPtr(const void* data) const override;
  bool hasPrimaryContext(DeviceIndex device_index) const override;
  DeviceIndex deviceCount() const override;
  void setCurrentDevice(DeviceIndex device) const override;
  DeviceIndex getCurrentDevice() const override;
};

} // namespace at::xpu::detail
