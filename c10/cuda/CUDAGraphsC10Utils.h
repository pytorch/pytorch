#pragma once

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAStream.h>

#include <iostream>
#include <optional>

// CUDA Graphs utils used by c10 and aten.
// aten/cuda/CUDAGraphsUtils.cuh adds utils used by aten only.

namespace c10::cuda {

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

// Returns the capture status of the given stream (or current stream if not
// specified). May initialize a CUDA context if called with the default stream
// argument and no context exists yet.
inline CaptureStatus currentStreamCaptureStatusMayInitCtx(
    std::optional<cudaStream_t> stream = std::nullopt) {
  cudaStream_t s = stream.value_or(
      static_cast<cudaStream_t>(c10::cuda::getCurrentCUDAStream()));
  cudaStreamCaptureStatus status{cudaStreamCaptureStatusNone};
  C10_CUDA_CHECK(cudaStreamIsCapturing(s, &status));
  return CaptureStatus(status);
}

inline bool isStreamCapturingMayInitCtx(
    std::optional<cudaStream_t> stream = std::nullopt) {
  return currentStreamCaptureStatusMayInitCtx(stream) == CaptureStatus::Active;
}

inline std::optional<CaptureId_t> currentStreamCaptureIdMayInitCtx(
    std::optional<cudaStream_t> stream = std::nullopt) {
  cudaStream_t s = stream.value_or(
      static_cast<cudaStream_t>(c10::cuda::getCurrentCUDAStream()));
  cudaStreamCaptureStatus status{};
  CaptureId_t capture_id = 0;
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(s, &status, &capture_id));
  if (status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive) {
    return capture_id;
  }
  return std::nullopt;
}

} // namespace c10::cuda
