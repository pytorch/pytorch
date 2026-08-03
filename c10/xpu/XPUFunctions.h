#pragma once

#include <c10/core/Device.h>
#include <c10/xpu/XPUDeviceProp.h>
#include <c10/xpu/XPUMacros.h>

// The naming convention used here matches the naming convention of torch.xpu

namespace c10::xpu {

// Log a warning only once if no devices are detected.
C10_XPU_API DeviceIndex device_count();

// Throws an error if no devices are detected.
C10_XPU_API DeviceIndex device_count_ensure_non_zero();

C10_XPU_API DeviceIndex current_device();

C10_XPU_API void set_device(DeviceIndex device);

C10_XPU_API DeviceIndex exchange_device(DeviceIndex device);

C10_XPU_API DeviceIndex maybe_exchange_device(DeviceIndex to_device);

C10_XPU_API sycl::device& get_raw_device(DeviceIndex device);

C10_XPU_API sycl::context& get_device_context();

C10_XPU_API void get_device_properties(
    DeviceProp* device_prop,
    DeviceIndex device);

C10_XPU_API DeviceIndex get_device_idx_from_pointer(void* ptr);

static inline void check_device_index(DeviceIndex device_index) {
  TORCH_CHECK(
      device_index >= 0 && device_index < c10::xpu::device_count(),
      "The device index is out of range. It must be in [0, ",
      static_cast<int>(c10::xpu::device_count()),
      "), but got ",
      static_cast<int>(device_index),
      ".");
}

} // namespace c10::xpu
