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

using CaptureId_t = c10::CaptureId_t;
using CaptureStatus = c10::cuda::CaptureStatus;

// Use this version where you don't want to create a CUDA context if none exists.
inline CaptureStatus currentStreamCaptureStatus() {
  // don't create a context if we don't have to
  if (c10::cuda::hasPrimaryContext(c10::cuda::current_device())) {
    return c10::cuda::currentStreamCaptureStatusMayInitCtx();
  } else {
    return CaptureStatus::None;
  }
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
