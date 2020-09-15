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

// When the global flag `allow_tf32` set to true, cuBLAS handles are
// automatically configured to use math mode CUBLAS_TF32_TENSOR_OP_MATH.
// For some operators, such as addmv, TF32 has no performance improves
// but causes precision loss. To help this case, this class implements
// a RAII guard that can be used to quickly disable TF32 within its scope.
//
// Usage:
//     cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
//     NoTF32Guard disable_tf32(handle);
struct NoTF32Guard {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  cublasHandle_t& handle;
  cublasMath_t original_mode;
  NoTF32Guard(cublasHandle_t& handle): handle(handle) {
    TORCH_CUDABLAS_CHECK(cublasGetMathMode(handle, &original_mode));
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  }
  ~NoTF32Guard() {
    cublasSetMathMode(handle, original_mode);
  }
#else
  NoTF32Guard(cublasHandle_t& handle){}
#endif
};

}} // namespace at::cuda
