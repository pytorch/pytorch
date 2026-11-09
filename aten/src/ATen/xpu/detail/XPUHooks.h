#pragma once

#include <ATen/detail/XPUHooksInterface.h>

namespace at::xpu::detail {

// The real implementation of XPUHooksInterface
struct XPUHooks : public at::XPUHooksInterface {
  XPUHooks() {}
  void init() const override;
  bool hasXPU() const override;
  std::string showConfig() const override;
  C10_DEPRECATED_MESSAGE(
      "XPUHooks::getGlobalIdxFromDevice() is deprecated since 2.16 and will be removed in 2.18. The SYCL index within the platform equals the torch device index, so use the torch device index instead. See https://github.com/pytorch/pytorch/issues/204019")
  int32_t getGlobalIdxFromDevice(const at::Device& device) const override;
  const Generator& getDefaultGenerator(
      DeviceIndex device_index = -1) const override;
  Generator getNewGenerator(DeviceIndex device_index = -1) const override;
  Device getDeviceFromPtr(void* data) const override;
  c10::DeviceIndex getNumGPUs() const override;
  DeviceIndex current_device() const override;
  void deviceSynchronize(DeviceIndex device_index) const override;
  Allocator* getPinnedMemoryAllocator() const override;

  bool isBuilt() const override {
    return true;
  }
  bool isAvailable() const override;
  bool isPinnedPtr(const void* data) const override;
  bool hasPrimaryContext(DeviceIndex device_index) const override;
  DeviceIndex deviceCount() const override;
  DeviceIndex getCurrentDevice() const override;
  const at::xpu::LevelZero& level_zero() const override;
};

} // namespace at::xpu::detail
