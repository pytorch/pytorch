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
  Generator getXPUGenerator(DeviceIndex device_index = -1) const override;
  const Generator& getDefaultXPUGenerator(
      DeviceIndex device_index = -1) const override;
  Device getDeviceFromPtr(void* data) const override;
  c10::DeviceIndex getNumGPUs() const override;
  DeviceIndex current_device() const override;
  void deviceSynchronize(DeviceIndex device_index) const override;
  Allocator* getPinnedMemoryAllocator() const override;
  bool isPinnedPtr(const void* data) const override;
  bool hasPrimaryContext(DeviceIndex device_index) const override;
  DeviceIndex deviceCount() const override;
  DeviceIndex getCurrentDevice() const override;
  c10::Stream getCurrentStream(DeviceIndex device) const override;
  void setCurrentStream(const c10::Stream& stream) override;
};

} // namespace at::xpu::detail
