#pragma once

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAStream.h>

#include <iostream>
#include <memory>
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

// Use this version where you're sure a CUDA context exists already.
inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
  cudaStreamCaptureStatus status{cudaStreamCaptureStatusNone};
  C10_CUDA_CHECK(
      cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream(), &status));
  return CaptureStatus(status);
}

inline CaptureStatus captureStatusMayInitCtx(cudaStream_t stream) {
  cudaStreamCaptureStatus status{cudaStreamCaptureStatusNone};
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
  return CaptureStatus(status);
}

inline bool isStreamCapturingMayInitCtx(cudaStream_t stream) {
  return captureStatusMayInitCtx(stream) == CaptureStatus::Active;
}

struct CaptureInfo {
  CaptureStatus status;
  CaptureId_t id;
  cudaGraph_t graph;
};

inline CaptureInfo captureInfoMayInitCtx(cudaStream_t stream) {
  cudaStreamCaptureStatus status{};
  CaptureId_t capture_id = 0;
  cudaGraph_t graph = nullptr;
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 13000)
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(
      stream, &status, &capture_id, &graph, nullptr, nullptr, nullptr));
#else
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo_v2(
      stream, &status, &capture_id, &graph, nullptr, nullptr));
#endif
  return {CaptureStatus(status), capture_id, graph};
}

template <typename T>
void retainGraphUserObject(
    cudaGraph_t graph,
    std::unique_ptr<T> data,
    cudaHostFn_t destroy) {
  cudaUserObject_t user_object{};
  C10_CUDA_CHECK(cudaUserObjectCreate(
      &user_object, data.get(), destroy, 1, cudaUserObjectNoDestructorSync));
  data.release();

  auto status =
      cudaGraphRetainUserObject(graph, user_object, 1, cudaGraphUserObjectMove);
  if (status != cudaSuccess) {
    C10_CUDA_CHECK_WARN(cudaUserObjectRelease(user_object, 1));
    C10_CUDA_CHECK(status);
  }
}

inline std::optional<CaptureId_t> currentStreamCaptureIdMayInitCtx() {
  auto info = captureInfoMayInitCtx(c10::cuda::getCurrentCUDAStream());
  if (info.status == CaptureStatus::Active) {
    return info.id;
  }
  return std::nullopt;
}

inline std::optional<CaptureId_t> captureIdMayInitCtx(cudaStream_t stream) {
  auto info = captureInfoMayInitCtx(stream);
  if (info.status == CaptureStatus::Active) {
    return info.id;
  }
  return std::nullopt;
}

} // namespace c10::cuda
