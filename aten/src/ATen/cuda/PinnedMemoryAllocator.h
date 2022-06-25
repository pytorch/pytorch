#pragma once

#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/core/Allocator.h>

namespace at {
namespace cuda {

inline TORCH_CUDA_CPP_API at::Allocator* getPinnedMemoryAllocator() {
  return getCachingHostAllocator();
}
} // namespace cuda
} // namespace at
