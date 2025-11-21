#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/xpu/XPUStream.h>

namespace c10::xpu::XPUCachingAllocator {

C10_XPU_API Allocator* get();

C10_XPU_API void init(DeviceIndex device_count);

C10_XPU_API void emptyCache();

C10_XPU_API void resetPeakStats(DeviceIndex device);

C10_XPU_API void resetAccumulatedStats(DeviceIndex device);

C10_XPU_API c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
    DeviceIndex device);

C10_XPU_API void* raw_alloc(size_t size);

C10_XPU_API void raw_delete(void* ptr);

C10_XPU_API void recordStream(const DataPtr& dataPtr, XPUStream stream);

C10_XPU_API void enablePeerAccess(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access);

C10_XPU_API double getMemoryFraction(DeviceIndex device);

C10_XPU_API void setMemoryFraction(double fraction, DeviceIndex device);

} // namespace c10::xpu::XPUCachingAllocator
