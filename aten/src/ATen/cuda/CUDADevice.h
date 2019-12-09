#pragma once

#include <ATen/cuda/Exceptions.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace at {
namespace cuda {

inline Device getDeviceFromPtr(void* ptr) {
  cudaPointerAttributes attr;
  AT_CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
  return {DeviceType::CUDA, static_cast<int16_t>(attr.device)};
}

}} // namespace at::cuda
