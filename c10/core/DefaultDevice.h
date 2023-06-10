#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Export.h>

namespace c10 {
C10_API void set_default_argument_device_type(c10::DeviceType device_type);
C10_API c10::DeviceType get_default_argument_device_type();
} // namespace c10
