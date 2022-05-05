#pragma once

#include <ATen/cuda/Exceptions.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace at {
namespace cuda {

inline Device getDeviceFromPtr(void* ptr) {
  cudaPointerAttributes attr{};

  AT_CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  TORCH_CHECK(attr.type != cudaMemoryTypeUnregistered,
    "The specified pointer resides on host memory and is not registered with any CUDA device.");
#endif

  return {DeviceType::CUDA, static_cast<DeviceIndex>(attr.device)};
}

}} // namespace at::cuda
