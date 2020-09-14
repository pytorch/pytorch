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

struct NoTF32Guard {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  cublasHandle_t &handle;
  cublasMath_t original_mode;
  NoTF32Guard(cublasHandle_t &handle): handle(handle) {
    TORCH_CUDABLAS_CHECK(cublasGetMathMode(handle, &original_mode));
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  }
  ~NoTF32Guard() {
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, original_mode));
  }
#endif
};

}} // namespace at::cuda
