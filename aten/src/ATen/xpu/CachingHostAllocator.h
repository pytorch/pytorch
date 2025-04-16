#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/xpu/XPUEvent.h>
#include <c10/core/Allocator.h>
#include <c10/xpu/XPUStream.h>

namespace at::xpu {

inline TORCH_XPU_API at::HostAllocator* getCachingHostAllocator() {
  return at::getHostAllocator(at::kXPU);
}

inline TORCH_XPU_API bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::xpu::XPUStream stream) {
  return getCachingHostAllocator()->record_event(ptr, ctx, stream.unwrap);
}

inline TORCH_XPU_API void CachingHostAllocator_emptyCache() {
  getCachingHostAllocator()->empty_cache();
}

inline TORCH_XPU_API at::DataPtr HostAlloc(size_t size) {
  return getCachingHostAllocator()->allocate(size);
}

} // namespace at::xpu
