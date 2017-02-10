/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "gloo/common/logging.h"

namespace gloo {

#define CUDA_CHECK(condition)                   \
  do {                                          \
    cudaError_t error = condition;              \
    GLOO_ENFORCE_EQ(                            \
      error,                                    \
      cudaSuccess,                              \
      "Error at: ",                             \
      __FILE__,                                 \
      ":",                                      \
      __LINE__,                                 \
      ": ",                                     \
      cudaGetErrorString(error));               \
  } while (0)

inline int getCurrentGPUID() {
  int id = 0;
  CUDA_CHECK(cudaGetDevice(&id));
  return id;
}

inline int getGPUIDForPointer(const void* ptr) {
  cudaPointerAttributes attr;
  CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
  return attr.device;
}

class CudaDeviceGuard {
 public:
  CudaDeviceGuard() : previous_(getCurrentGPUID()) {
  }

  ~CudaDeviceGuard() {
    CUDA_CHECK(cudaSetDevice(previous_));
  }

 private:
  int previous_;
};

} // namespace gloo
