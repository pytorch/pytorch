#pragma once

#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <utility>

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

// Use this version where you're sure a CUDA context exists already.
inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
  cudaStreamCaptureStatus is_capturing{cudaStreamCaptureStatusNone};
  C10_CUDA_CHECK(
      cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream(), &is_capturing));
  return CaptureStatus(is_capturing);
}

// this is a wrapper around cudaStreamGetCaptureInfo(). It returns
// std::nullopt if the stream is not capturing. Otherwise, it returns
// an array of pointers to the current terminal nodes in stream
// capture and the size of that array. Be forewarned that "The array
// pointer is valid until the next API call which operates on the
// stream or until the capture is terminated."
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g8d9312f1098c45e2ed43c949cfccf1f7
inline std::optional<std::tuple<const cudaGraphNode_t*, size_t>>
streamGetTerminalNodes(CUDAStream stream) {
  cudaStreamCaptureStatus status{cudaStreamCaptureStatusNone};
  unsigned long long capture_id{0};
  cudaGraph_t graph{};
  const cudaGraphNode_t* terminals{nullptr};
  size_t num_terminals{0};

#if (defined(CUDA_VERSION) && CUDA_VERSION >= 13000)
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(
      stream,
      &status,
      &capture_id,
      &graph,
      &terminals,
      nullptr,
      &num_terminals));
#else
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo_v2(
      stream, &status, &capture_id, &graph, &terminals, &num_terminals));
#endif

  if (status == cudaStreamCaptureStatusNone) {
    return std::nullopt;
  } else {
    return std::make_tuple(terminals, num_terminals);
  }
}

} // namespace c10::cuda
