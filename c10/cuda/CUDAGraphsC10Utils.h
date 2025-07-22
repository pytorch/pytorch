#pragma once

#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <utility>

// CUDA Graphs utils used by c10 and aten.
// aten/cuda/CUDAGraphsUtils.cuh adds utils used by aten only.

namespace c10::cuda {

using CaptureId_t = unsigned long long;

// first is set if the instance is created by CUDAGraph::capture_begin.
// second is set if the instance is created by at::cuda::graph_pool_handle.
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

// RAII guard for "cudaStreamCaptureMode", a thread-local value
// that controls the error-checking strictness of a capture.
struct C10_CUDA_API CUDAStreamCaptureModeGuard {
  CUDAStreamCaptureModeGuard(cudaStreamCaptureMode desired)
      : strictness_(desired) {
    C10_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&strictness_));
  }
  CUDAStreamCaptureModeGuard(const CUDAStreamCaptureModeGuard&) = delete;
  CUDAStreamCaptureModeGuard(CUDAStreamCaptureModeGuard&&) = delete;
  CUDAStreamCaptureModeGuard& operator=(const CUDAStreamCaptureModeGuard&) =
      delete;
  CUDAStreamCaptureModeGuard& operator=(CUDAStreamCaptureModeGuard&&) = delete;
  ~CUDAStreamCaptureModeGuard() {
    C10_CUDA_CHECK_WARN(cudaThreadExchangeStreamCaptureMode(&strictness_));
  }

 private:
  cudaStreamCaptureMode strictness_;
};

// Protects against enum cudaStreamCaptureStatus implementation changes.
// Some compilers seem not to like static_assert without the messages.
static_assert(
    int(cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) == 0,
    "unexpected int(cudaStreamCaptureStatusNone) value");
static_assert(
    int(cudaStreamCaptureStatus::cudaStreamCaptureStatusActive) == 1,
    "unexpected int(cudaStreamCaptureStatusActive) value");
static_assert(
    int(cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated) == 2,
    "unexpected int(cudaStreamCaptureStatusInvalidated) value");

enum class CaptureStatus : int {
  None = int(cudaStreamCaptureStatus::cudaStreamCaptureStatusNone),
  Active = int(cudaStreamCaptureStatus::cudaStreamCaptureStatusActive),
  Invalidated = int(cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated)
};

inline std::ostream& operator<<(std::ostream& os, CaptureStatus status) {
  switch (status) {
    case CaptureStatus::None:
      os << "cudaStreamCaptureStatusNone";
      break;
    case CaptureStatus::Active:
      os << "cudaStreamCaptureStatusActive";
      break;
    case CaptureStatus::Invalidated:
      os << "cudaStreamCaptureStatusInvalidated";
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Unknown CUDA graph CaptureStatus", int(status));
  }
  return os;
}

// Use this version where you're sure a CUDA context exists already.
inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
  cudaStreamCaptureStatus is_capturing{cudaStreamCaptureStatusNone};
  C10_CUDA_CHECK(
      cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream(), &is_capturing));
  return CaptureStatus(is_capturing);
}

} // namespace c10::cuda
