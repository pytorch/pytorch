#pragma once

#ifdef _WIN32
  #ifdef OPENREG_EXPORTS
    #define OPENREG_API __declspec(dllexport)
  #else
    #define OPENREG_API __declspec(dllimport)
  #endif
#else
  #define OPENREG_API
#endif

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

#include <limits>

namespace c10::openreg {

OPENREG_API c10::DeviceIndex device_count() noexcept;
OPENREG_API c10::DeviceIndex current_device();
OPENREG_API void set_device(c10::DeviceIndex device);

OPENREG_API DeviceIndex ExchangeDevice(DeviceIndex device);

} // namespace c10::openreg
