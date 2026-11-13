/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "DeviceUtil.h"

#include <mutex>

namespace KINETO_NAMESPACE {

bool amdGpuAvailable = false;
bool cudaGpuAvailable = false;

bool isAMDGpuAvailable() {
#ifdef HAS_ROCTRACER
  static std::once_flag once;
  std::call_once(once, [] {
    // determine AMD GPU availability on the system
    hipError_t error;
    int deviceCount;
    error = hipGetDeviceCount(&deviceCount);
    amdGpuAvailable = (error == hipSuccess && deviceCount > 0);
  });
#endif
  return amdGpuAvailable;
}

bool isCUDAGpuAvailable() {
#ifdef HAS_CUPTI
  static std::once_flag once;
  std::call_once(once, [] {
    // determine CUDA GPU availability on the system
    cudaError_t error;
    int deviceCount;
    error = cudaGetDeviceCount(&deviceCount);
    cudaGpuAvailable = (error == cudaSuccess && deviceCount > 0);
  });
#endif
  return cudaGpuAvailable;
}

bool isGpuAvailable() {
  bool amd = isAMDGpuAvailable();
  bool cuda = isCUDAGpuAvailable();
  return amd || cuda;
}

} // namespace KINETO_NAMESPACE
