#pragma once

#ifdef _WIN32
  #define OPENREG_EXPORT __declspec(dllexport)
#else
  #define OPENREG_EXPORT __attribute__((visibility("default")))
#endif

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

#include <limits>

namespace c10::openreg {

OPENREG_EXPORT c10::DeviceIndex device_count() noexcept;
OPENREG_EXPORT c10::DeviceIndex current_device();
OPENREG_EXPORT void set_device(c10::DeviceIndex device);

OPENREG_EXPORT DeviceIndex ExchangeDevice(DeviceIndex device);

} // namespace c10::openreg
