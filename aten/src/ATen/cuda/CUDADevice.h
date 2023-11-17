#pragma once

#include <ATen/cuda/Exceptions.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace at::cuda {

inline Device getDeviceFromPtr(void* ptr) {
  cudaPointerAttributes attr{};

  AT_CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));

#if !defined(USE_ROCM)
  TORCH_CHECK(attr.type != cudaMemoryTypeUnregistered,
    "The specified pointer resides on host memory and is not registered with any CUDA device.");
#endif

  return {c10::DeviceType::CUDA, static_cast<DeviceIndex>(attr.device)};
}

} // namespace at::cuda
