#pragma once

// This header provides C++ wrappers around commonly used CUDA API functions.
// The benefit of using C++ here is that we can raise an exception in the
// event of an error, rather than explicitly pass around error codes.  This
// leads to more natural APIs.
//
// The naming convention used here matches the naming convention of torch.cuda

#include <cuda_runtime_api.h>

#include <c10/macros/Macros.h>
#include <c10/Device.h>

namespace c10 {
namespace cuda {

inline DeviceIndex device_count() {
  int count;
  C10_CUDA_CHECK(cudaGetDeviceCount(&count));
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
