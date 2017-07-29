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

inline int getDeviceCount() {
  int count;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  return count;
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
// Convenience class used for tests and benchmarks.
template<typename T>
class CudaMemory {
 public:
  explicit CudaMemory(size_t elements);
  CudaMemory(CudaMemory&&) noexcept;
  ~CudaMemory();

  void set(int val, size_t stride = 0, cudaStream_t stream = kStreamNotSet);

  T* operator*() const {
    return ptr_;
  }

  std::unique_ptr<T[]> copyToHost() const;

  const size_t elements;
  const size_t bytes;

 protected:
  CudaMemory(const CudaMemory&) = delete;
  CudaMemory& operator=(const CudaMemory&) = delete;

  int device_;
  T* ptr_;
};

// Container class for a set of per-device streams
class CudaDeviceStreams {
 public:
  CudaDeviceStreams() {
    const int numDevices = getDeviceCount();
    streams_.reserve(numDevices);
    for (auto i = 0; i < numDevices; i++) {
      streams_.push_back(CudaStream(i));
    }
  }
  cudaStream_t operator[](const int i) {
    GLOO_ENFORCE_LT(i, streams_.size());
    return *streams_[i];
  }

 protected:
  CudaDeviceStreams(const CudaDeviceStreams&) = delete;
  CudaDeviceStreams& operator=(const CudaDeviceStreams&) = delete;

  std::vector<CudaStream> streams_;
};

} // namespace gloo
