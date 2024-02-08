#pragma once

#include <ATen/Context.h>
#include <c10/xpu/XPUFunctions.h>

namespace at::xpu {

inline Device getDeviceFromPtr(void* ptr) {
  auto device = c10::xpu::get_device_idx_from_pointer(ptr);
  return {c10::DeviceType::XPU, static_cast<DeviceIndex>(device)};
}

} // namespace at::xpu
