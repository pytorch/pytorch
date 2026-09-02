#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/Allocator.h>
#include <c10/macros/Export.h>

namespace at::xpu {

inline TORCH_XPU_API at::HostAllocator* getPinnedMemoryAllocator() {
  return at::getHostAllocator(at::kXPU);
}
} // namespace at::xpu
