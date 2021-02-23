#pragma once

// CUDA Graphs utils used by c10 and aten.
// aten/cuda/CUDAGraphsAtenUtils.cuh has utils used by aten only.

namespace c10 {
namespace cuda {

using CaptureId_t = unsigned long long;

// RAII guard for "cudaStreamCaptureMode", a thread-local value
// that controls the error-checking strictness of a capture.
#if CUDA_VERSION >= 11000
struct TORCH_CUDA_CPP_API CUDAStreamCaptureModeGuard {
  CUDAStreamCaptureModeGuard(cudaStreamCaptureMode desired) {
    strictness_ = desired;
    C10_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&strictness_));
  }
  ~CUDAStreamCaptureModeGuard() {
    C10_CUDA_CHECK_WARN(cudaThreadExchangeStreamCaptureMode(&strictness_));
  }

  private:
  cudaStreamCaptureMode strictness_;
};
#endif

} // namespace c10
} // namespace cuda
