#pragma once

#include <c10/detail/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

namespace at {
namespace detail {

struct CPUGuardImpl final : public c10::detail::DeviceGuardImplInterface {
  CPUGuardImpl() {}
  DeviceType type() const override {
    return DeviceType::CPU;
  }
  Device exchangeDevice(Device) const override {
    // no-op
    return Device(DeviceType::CPU, -1);

  }
  Device getDevice() const override {
    return Device(DeviceType::CPU, -1);
  }
  void setDevice(Device) const override {
    // no-op
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    // no-op
  }
  Stream getStream(Device d) const noexcept override {
    // no-op
    return Stream(Device(DeviceType::CPU, -1), 0);
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const noexcept override {
    // no-op
    return Stream(Device(DeviceType::CPU, -1), 0);
  }
};

}} // namespace at::detail
