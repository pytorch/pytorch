#pragma once

#include <c10/core/Device.h>
#include <c10/cuda/PeerToPeerAccess.h>
#include <c10/macros/Macros.h>

#include <cstdint>

namespace at::cuda {

namespace detail {

/// Initialize the peer-to-peer and fabric access caches.
/// Forwards to c10::cuda::detail::init_p2p_access_cache.
/// @param num_devices The number of CUDA devices in the system.
inline void init_p2p_access_cache(int64_t num_devices) {
  c10::cuda::detail::init_p2p_access_cache(num_devices);
}

} // namespace detail

/// Query if peer-to-peer access is available between two devices.
/// This wrapper ensures CUDA lazy initialization before forwarding to c10.
/// @param source_dev The source device index.
/// @param dest_dev The destination device index.
/// @return true if P2P access is available, false otherwise.
TORCH_CUDA_CPP_API bool get_p2p_access(
    c10::DeviceIndex source_dev,
    c10::DeviceIndex dest_dev);

/// Query if GPU fabric (high-speed interconnect) is available for a device.
/// This wrapper ensures CUDA lazy initialization before forwarding to c10.
/// @param device The device index to check.
/// @return true if fabric access is available, false otherwise.
TORCH_CUDA_CPP_API bool get_fabric_access(c10::DeviceIndex device);

} // namespace at::cuda
