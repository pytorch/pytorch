#pragma once

#include <ATen/accelerator/Accelerator.h>
#include <ATen/core/CachingHostAllocator.h>

namespace at::accelerator {

// Releases all unused device memory currently held by the accelerator's
// device-side caching allocator. The freed memory becomes available for reuse
// by other applications or processes.
void emptyCache() {
  const auto device_type = getAccelerator(true).value();
  at::getDeviceAllocator(device_type)->emptyCache();
}

// Releases all unused host (pinned) memory currently held by the accelerator's
// host-side caching allocator. The freed memory becomes available for reuse by
// other applications or processes.
void emptyHostCache() {
  const auto device_type = getAccelerator(true).value();
  at::getHostAllocator(device_type)->empty_cache();
}

at::CachingDeviceAllocator::DeviceStats getDeviceStats(
    c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  return at::getDeviceAllocator(device_type)->getDeviceStats(device_index);
}

void resetAccumulatedStats(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  at::getDeviceAllocator(device_type)->resetAccumulatedStats(device_index);
}

void resetPeakStats(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  at::getDeviceAllocator(device_type)->resetPeakStats(device_index);
}

std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  return at::getDeviceAllocator(device_type)->getMemoryInfo(device_index);
}
} // namespace at::accelerator
