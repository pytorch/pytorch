#pragma once

#include <c10/core/CachingDeviceAllocator.h>

namespace at::accelerator {

// Releases all unused device memory currently held by the accelerator's
// device-side caching allocator. The freed memory becomes available for reuse
// by other applications or processes.
TORCH_API void emptyCache();

// Releases all unused host (pinned) memory currently held by the accelerator's
// host-side caching allocator. The freed memory becomes available for reuse by
// other applications or processes.
TORCH_API void emptyHostCache();

TORCH_API at::CachingDeviceAllocator::DeviceStats getDeviceStats(
    c10::DeviceIndex device_index);

TORCH_API void resetAccumulatedStats(c10::DeviceIndex device_index);

TORCH_API void resetPeakStats(c10::DeviceIndex device_index);

TORCH_API std::pair<size_t, size_t> getMemoryInfo(
    c10::DeviceIndex device_index);

} // namespace at::accelerator
