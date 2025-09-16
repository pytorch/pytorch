#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

#include <include/Macros.h>

#include <limits>

namespace c10::openreg {

OPENREG_EXPORT c10::DeviceIndex device_count() noexcept;
OPENREG_EXPORT c10::DeviceIndex current_device();
OPENREG_EXPORT void set_device(c10::DeviceIndex device);

OPENREG_EXPORT DeviceIndex ExchangeDevice(DeviceIndex device);

} // namespace c10::openreg
