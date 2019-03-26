#pragma once

// This header provides C++ wrappers around commonly used CUDA API functions.
// The benefit of using C++ here is that we can raise an exception in the
// event of an error, rather than explicitly pass around error codes.  This
// leads to more natural APIs.
//
// The naming convention used here matches the naming convention of torch.cuda

#include <cuda_runtime_api.h>

#include <c10/macros/Macros.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAException.h>

namespace c10 {
namespace cuda {

inline DeviceIndex device_count() noexcept {
  int count;
  // NB: In the past, we were inconsistent about whether or not this reported
  // an error if there were driver problems are not.  Based on experience
  // interacting with users, it seems that people basically ~never want this
  // function to fail; it should just return zero if things are not working.
  // Oblige them.
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    // Clear out the error state, so we don't spuriously trigger someone else.
    // (This shouldn't really matter, since we won't be running very much CUDA
    // code in this regime.)
    cudaError_t last_err = cudaGetLastError();
    (void)last_err;
    return 0;
  }
  return static_cast<DeviceIndex>(count);
}

inline DeviceIndex current_device() {
  int cur_device;
  C10_CUDA_CHECK(cudaGetDevice(&cur_device));
  return static_cast<DeviceIndex>(cur_device);
}

inline void set_device(DeviceIndex device) {
  C10_CUDA_CHECK(cudaSetDevice(static_cast<int>(device)));
}

}} // namespace c10::cuda
