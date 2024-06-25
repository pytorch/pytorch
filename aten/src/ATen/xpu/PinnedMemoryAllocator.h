#pragma once

#include <ATen/xpu/CachingHostAllocator.h>
#include <c10/core/Allocator.h>

namespace at::xpu {

inline TORCH_XPU_API at::Allocator* getPinnedMemoryAllocator() {
  return getCachingHostAllocator();
}
} // namespace at::xpu
