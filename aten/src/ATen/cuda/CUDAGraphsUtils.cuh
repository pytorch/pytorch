#pragma once

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <ATen/cuda/detail/CUDAHooks.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>

// c10/cuda/CUDAGraphsC10Utils.h has utils used by both c10 and aten.
// This file adds utils used by aten only.

namespace at::cuda {

using CaptureId_t = c10::cuda::CaptureId_t;
using CaptureStatus = c10::cuda::CaptureStatus;

// Returns the capture status of the given stream (or current stream if not specified).
// Safe version that checks for existing context before potentially creating one.
// If no stream is provided and no context exists, returns CaptureStatus::None.
inline CaptureStatus currentStreamCaptureStatus(
    std::optional<cudaStream_t> stream = std::nullopt) {
  // If stream is explicitly provided, it's safe to query it directly
  if (stream.has_value()) {
    return c10::cuda::currentStreamCaptureStatusMayInitCtx(stream);
  }
  // Otherwise, check if context exists before using current stream
  if (c10::cuda::hasPrimaryContext(c10::cuda::current_device())) {
    return c10::cuda::currentStreamCaptureStatusMayInitCtx(stream);
  }
  return CaptureStatus::None;
}

inline bool isStreamCapturing(
    std::optional<cudaStream_t> stream = std::nullopt) {
  return currentStreamCaptureStatus(stream) == CaptureStatus::Active;
}

inline std::optional<CaptureId_t> currentStreamCaptureId(
    std::optional<cudaStream_t> stream = std::nullopt) {
  // If stream is explicitly provided, it's safe to query it directly
  if (stream.has_value()) {
    return c10::cuda::currentStreamCaptureIdMayInitCtx(stream);
  }
  // Otherwise, check if context exists before using current stream
  if (c10::cuda::hasPrimaryContext(c10::cuda::current_device())) {
    return c10::cuda::currentStreamCaptureIdMayInitCtx(stream);
  }
  return std::nullopt;
}

inline void assertNotCapturing(const std::string& attempt) {
  auto status = currentStreamCaptureStatus();
  TORCH_CHECK(status == CaptureStatus::None,
              attempt,
              " during CUDA graph capture. If you need this call to be captured, "
              "please file an issue. "
              "Current cudaStreamCaptureStatus: ",
              status);
}

inline void errorIfCapturingCudnnBenchmark(const std::string& version_specific) {
  auto status = currentStreamCaptureStatus();
  TORCH_CHECK(status == CaptureStatus::None,
              "Current cudaStreamCaptureStatus: ",
              status,
              "\nCapturing ",
              version_specific,
              "is prohibited. Possible causes of this error:\n"
              "1. No warmup iterations occurred before capture.\n"
              "2. The convolutions you're trying to capture use dynamic shapes, "
              "in which case capturing them is generally prohibited.");
}

} // namespace at::cuda
