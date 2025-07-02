#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include "../backend/include/openreg.h"

#include <mutex>
#include <optional>

namespace c10::backend {

c10::DeviceIndex device_count() noexcept;
DeviceIndex current_device();
void set_device(c10::DeviceIndex device);

DeviceIndex ExchangeDevice(DeviceIndex device);

} // namespace c10::backend
