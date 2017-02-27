/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <memory>
#include <mutex>

#include "gloo/common/logging.h"
#include "gloo/cuda.h"

namespace gloo {

// Default mutex to synchronize contentious CUDA and NCCL operations
extern std::mutex gCudaMutex;

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

class CudaDeviceScope {
 public:
  explicit CudaDeviceScope(int device) : guard_() {
    CUDA_CHECK(cudaSetDevice(device));
  }

 private:
  CudaDeviceGuard guard_;
};

// Managed chunk of GPU memory.
// Convience class used for tests and benchmarks.
template<typename T>
class CudaMemory {
 public:
  explicit CudaMemory(size_t n);
  CudaMemory(CudaMemory&&) noexcept;
  ~CudaMemory();

  void set(T val, cudaStream_t stream = kStreamNotSet);

  T* operator*() const {
    return ptr_;
  }

  std::unique_ptr<T[]> copyToHost();

 protected:
  CudaMemory(const CudaMemory&) = delete;
  CudaMemory& operator=(const CudaMemory&) = delete;

  size_t n_;
  size_t bytes_;
  int device_;
  T* ptr_;
};

} // namespace gloo
