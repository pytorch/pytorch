#pragma once

#include <ATen/cuda/CachingHostAllocator.h>

namespace at::cuda {

C10_DEPRECATED_MESSAGE(
    "at::cuda::getPinnedMemoryAllocator() is deprecated. Please use at::getHostAllocator(at::kCUDA) instead.")
inline TORCH_CUDA_CPP_API at::HostAllocator* getPinnedMemoryAllocator() {
  return at::getHostAllocator(at::kCUDA);
}
} // namespace at::cuda
