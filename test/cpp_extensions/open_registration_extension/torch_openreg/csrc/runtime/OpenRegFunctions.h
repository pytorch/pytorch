#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

#include <limits>

namespace c10::openreg {

c10::DeviceIndex device_count() noexcept;
DeviceIndex current_device();
void set_device(c10::DeviceIndex device);

DeviceIndex ExchangeDevice(DeviceIndex device);

} // namespace c10::openreg
