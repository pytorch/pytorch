#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <c10/xpu/XPUMacros.h>

namespace c10::xpu {
namespace detail {
// Initialize the peer-to-peer access cache for XPU devices.
C10_XPU_API void init_p2p_access_cache(c10::DeviceIndex num_devices);
} // namespace detail

// Query if peer-to-peer access is available between two devices.
C10_XPU_API bool get_p2p_access(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access);

} // namespace c10::xpu
