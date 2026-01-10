#pragma once
#include <c10/hip/HIPCachingAllocator.h>

namespace c10 {
namespace hip {
// Wrapper for c10::reportMemoryUsageToProfiler that converts
// c10::DeviceType::HIP device to c10::DeviceType::CUDA. In ROCm builds, memory
// allocations are made on HIP devices. However, the PyTorch profiler records
// and associates TensorMetadata based on DeviceType::CUDA. Without this
// conversion, allocation events reported as HIP would not be matched with their
// corresponding usage in the profiler analysis.
inline void reportMemoryUsageToProfilerMasqueradingAsCUDA(
    void* ptr,
    int64_t alloc_size,
    size_t total_allocated,
    size_t total_reserved,
    Device device) {
  if (device.type() == c10::DeviceType::HIP)
    device = c10::Device(c10::DeviceType::CUDA, device.index());
  c10::reportMemoryUsageToProfiler(
      ptr, alloc_size, total_allocated, total_reserved, device);
}
} // namespace hip
} // namespace c10