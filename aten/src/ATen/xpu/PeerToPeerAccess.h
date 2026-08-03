#pragma once

#include <c10/xpu/PeerToPeerAccess.h>

#include <ATen/Context.h>

namespace at::xpu {
namespace detail {
// Initialize the peer-to-peer access cache for XPU devices.
inline void init_p2p_access_cache(c10::DeviceIndex num_devices) {
  c10::xpu::detail::init_p2p_access_cache(num_devices);
}
} // namespace detail

// Query if peer-to-peer access is available between two devices.
// This wrapper ensures XPU lazy initialization before forwarding to c10.
inline bool get_p2p_access(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access) {
  at::globalContext().lazyInitDevice(c10::DeviceType::XPU);
  return c10::xpu::get_p2p_access(dev, dev_to_access);
}

} // namespace at::xpu
