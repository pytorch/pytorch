#pragma once

#include <c10/core/Device.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/macros/Macros.h>

#include <cstdint>

namespace c10::cuda {

namespace detail {

/// Initialize the peer-to-peer and fabric access caches.
/// Must be called before any calls to get_p2p_access or get_fabric_access.
/// @param num_devices The number of CUDA devices in the system.
C10_CUDA_API void init_p2p_access_cache(int64_t num_devices);

} // namespace detail

/// Query if peer-to-peer access is available between two devices.
/// @param source_dev The source device index.
/// @param dest_dev The destination device index.
/// @return true if P2P access is available, false otherwise.
C10_CUDA_API bool get_p2p_access(
    c10::DeviceIndex source_dev,
    c10::DeviceIndex dest_dev);

/// Query if GPU fabric (high-speed interconnect like NVLink/NVSwitch) is
/// available for a device. This checks both hardware support and the ability
/// to allocate/export/import memory with fabric handles.
/// @param device The device index to check.
/// @return true if fabric access is available, false otherwise.
C10_CUDA_API bool get_fabric_access(c10::DeviceIndex device);

} // namespace c10::cuda
