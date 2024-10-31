#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/xpu/XPUEvent.h>
#include <c10/core/Allocator.h>
#include <c10/xpu/XPUStream.h>

namespace at::xpu {

TORCH_XPU_API c10::Allocator* getCachingHostAllocator();

TORCH_XPU_API bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::xpu::XPUStream stream);

TORCH_XPU_API void CachingHostAllocator_emptyCache();

inline TORCH_XPU_API at::DataPtr HostAlloc(size_t size) {
  return getCachingHostAllocator()->allocate(size);
}

} // namespace at::xpu
