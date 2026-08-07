/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>

#ifdef HAS_CUPTI
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

// ok to use fmt::format as error will not occur often. Can't use fmt::print
// easily since LOG(...) can return void, causes compiler error
#define CUDA_CALL(call)                                    \
  [&]() -> cudaError_t {                                   \
    cudaError_t _status_ = call;                           \
    if (_status_ != cudaSuccess) {                         \
      const char* _errstr_ = cudaGetErrorString(_status_); \
      LOG(WARNING) << fmt::format(                         \
          "function {} failed with error {} ({})",         \
          #call,                                           \
          _errstr_,                                        \
          (int)_status_);                                  \
    }                                                      \
    return _status_;                                       \
  }()

#define CUPTI_CALL(call)                           \
  [&]() -> CUptiResult {                           \
    CUptiResult _status_ = call;                   \
    if (_status_ != CUPTI_SUCCESS) {               \
      const char* _errstr_ = nullptr;              \
      cuptiGetResultString(_status_, &_errstr_);   \
      LOG(WARNING) << fmt::format(                 \
          "function {} failed with error {} ({})", \
          #call,                                   \
          _errstr_,                                \
          (int)_status_);                          \
    }                                              \
    return _status_;                               \
  }()

#elif defined(HAS_ROCTRACER)
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk/version.h>

#define CUDA_CALL(call)                                   \
  {                                                       \
    hipError_t _status_ = call;                           \
    if (_status_ != hipSuccess) {                         \
      const char* _errstr_ = hipGetErrorString(_status_); \
      LOG(WARNING) << fmt::format(                        \
          "function {} failed with error {} ({})",        \
          #call,                                          \
          _errstr_,                                       \
          (int)_status_);                                 \
    }                                                     \
  }

#define CUPTI_CALL(call) call

#else
#define CUPTI_CALL(call) call
#endif // HAS_CUPTI

#define CUPTI_CALL_NOWARN(call) call

namespace KINETO_NAMESPACE {

bool isAMDGpuAvailable();

bool isCUDAGpuAvailable();

bool isGpuAvailable();

} // namespace KINETO_NAMESPACE
