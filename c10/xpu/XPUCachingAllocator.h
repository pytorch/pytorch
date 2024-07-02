#pragma once

#include <c10/core/Allocator.h>
#include <c10/xpu/XPUStream.h>

namespace c10::xpu::XPUCachingAllocator {

struct Stat {
  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated = 0;
  int64_t freed = 0;
};

enum struct StatType : uint64_t {
  AGGREGATE = 0,
  SMALL_POOL = 1,
  LARGE_POOL = 2,
  NUM_TYPES = 3 // remember to update this whenever a new stat type is added
};

typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;

// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
  // SUM: bytes allocated by this memory allocator
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes requested by client code
  StatArray requested_bytes;
};

C10_XPU_API Allocator* get();

C10_XPU_API void init(DeviceIndex device_count);

C10_XPU_API void emptyCache();

C10_XPU_API void resetPeakStats(DeviceIndex device);

C10_XPU_API void resetAccumulatedStats(DeviceIndex device);

C10_XPU_API DeviceStats getDeviceStats(DeviceIndex device);

C10_XPU_API void* raw_alloc(size_t size);

C10_XPU_API void raw_delete(void* ptr);

C10_XPU_API void recordStream(const DataPtr& dataPtr, XPUStream stream);

} // namespace c10::xpu::XPUCachingAllocator
