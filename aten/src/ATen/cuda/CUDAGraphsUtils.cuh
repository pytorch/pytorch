#pragma once

#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/detail/UnpackRaw.cuh>
#include <ATen/cuda/detail/CUDAHooks.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/cuda/nccl.h>

// c10/cuda/CUDAGraphsC10Utils.h has utils used by both c10 and aten.
// This file adds utils used by aten only.

namespace at {
namespace cuda {

using CaptureId_t = c10::cuda::CaptureId_t;
using CaptureStatus = c10::cuda::CaptureStatus;

// Use this version where you don't want to create a CUDA context if none exists.
inline CaptureStatus currentStreamCaptureStatus() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  // don't create a context if we don't have to
  if (at::cuda::detail::hasPrimaryContext(c10::cuda::current_device())) {
    return c10::cuda::currentStreamCaptureStatusMayInitCtx();
  } else {
    return CaptureStatus::None;
  }
#else
  return CaptureStatus::None;
#endif
}

inline void assertNotCapturing(std::string attempt) {
  auto status = currentStreamCaptureStatus();
  TORCH_CHECK(status == CaptureStatus::None,
              attempt,
              " during CUDA graph capture. If you need this call to be captured, "
              "please file an issue. "
              "Current cudaStreamCaptureStatus: ",
              status);
}

inline void errorIfCapturingCudnnBenchmark(std::string version_specific) {
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

inline void errorIfCapturingNonCapturableNCCL() {
  auto status = currentStreamCaptureStatus();
  // parentheses avoid some compiler warnings
  static const uint64_t min_version = (((uint64_t)2) << 32) + (((uint64_t)9) << 16) + ((uint64_t)6);
  static const uint64_t cur_version = torch::cuda::nccl::version();
  if (cur_version < min_version) {
    TORCH_CHECK(status == CaptureStatus::None,
                "Capturing NCCL collectives is only allowed with NCCL >= 2.9.6");
  }
}

} // namespace cuda
} // namespace at
