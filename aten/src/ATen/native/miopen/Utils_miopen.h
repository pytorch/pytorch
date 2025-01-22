#pragma once

// TODO: Remove the condition on AT_ROCM_ENABLED entirely,
// don't build this file as part of CPU build.
#include <ATen/cuda/CUDAConfig.h>

#if AT_ROCM_ENABLED()

#include <c10/hip/HIPCachingAllocator.h>

namespace at { namespace native {

// Struct for managing GPU memory for HIP devices
struct GPUWorkspace {
  GPUWorkspace(size_t size) : size(size), data(NULL) {
    data = c10::hip::HIPCachingAllocator::raw_alloc(size);
  }
  GPUWorkspace(const GPUWorkspace&) = delete;
  GPUWorkspace(GPUWorkspace&&) = default;
  GPUWorkspace& operator=(GPUWorkspace&&) = default;
  ~GPUWorkspace() {
    if (data) {
      c10::hip::HIPCachingAllocator::raw_delete(data);
    }
  }

  size_t size;
  void* data;
};

}}  // namespace native

#endif