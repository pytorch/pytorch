#pragma once

#include <ATen/cuda/CUDAContext.h>

namespace at { namespace cuda {

// Check if every tensor in a list of tensors matches the current
// device.
inline bool check_device(ArrayRef<Tensor> ts) {
  if (ts.empty()) {
    return true;
  }
  Device curDevice = Device(kCUDA, current_device());
  for (const Tensor& t : ts) {
    if (t.device() != curDevice) return false;
  }
  return true;
}

}} // namespace at::cuda
