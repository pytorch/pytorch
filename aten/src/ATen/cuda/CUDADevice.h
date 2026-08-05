#pragma once

#include <ATen/cuda/Exceptions.h>
#include <c10/core/Device.h>

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

  TORCH_INTERNAL_ASSERT(
      attr.device >= 0 && attr.device < c10::Device::MAX_NUM_DEVICES,
      "cudaPointerGetAttributes returns invalid device ",
      attr.device);
  return {c10::DeviceType::CUDA, static_cast<DeviceIndex>(attr.device)};
}

} // namespace at::cuda
