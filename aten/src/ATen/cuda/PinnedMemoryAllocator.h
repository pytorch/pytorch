#pragma once

#include <c10/core/Allocator.h>
#include <ATen/cuda/CachingHostAllocator.h>

namespace at::cuda {

inline TORCH_CUDA_CPP_API at::Allocator* getPinnedMemoryAllocator() {
  return getCachingHostAllocator();
}
} // namespace at::cuda
