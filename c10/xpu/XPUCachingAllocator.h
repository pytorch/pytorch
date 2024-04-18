#pragma once

#include <c10/core/Allocator.h>
#include <c10/xpu/XPUStream.h>

namespace c10::xpu::XPUCachingAllocator {

C10_XPU_API Allocator* get();

C10_XPU_API void init(DeviceIndex device_count);

C10_XPU_API void emptyCache();

C10_XPU_API void* raw_alloc(size_t size);

C10_XPU_API void raw_delete(void* ptr);

C10_XPU_API void recordStream(const DataPtr& dataPtr, XPUStream stream);

} // namespace c10::xpu::XPUCachingAllocator
