/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <atomic>
#include <mutex>

#include <cuda.h>
#include <cuda_runtime.h>

namespace gloo {

extern const cudaStream_t kStreamNotSet;

class CudaShared {
 public:
  // Get the mutex used to synchronize CUDA and NCCL operations
  static std::mutex& getMutex() {
    return *mutex_;
  }

  // Set the mutex used to synchronize CUDA and NCCL operations
  static void setMutex(std::mutex* m) {
    mutex_ = m;
  }

 private:
  static std::atomic<std::mutex*> mutex_;
};

template<typename T>
class CudaDevicePointer {
 public:
  static CudaDevicePointer create(
    T* ptr,
    size_t count,
    cudaStream_t stream = kStreamNotSet);

  CudaDevicePointer(CudaDevicePointer&&) noexcept;
  ~CudaDevicePointer();

  T* operator*() const {
    return device_;
  }

  int getCount() const {
    return count_;
  }

  int getDeviceID() const {
    return deviceId_;
  }

  cudaStream_t getStream() const {
    return stream_;
  }

  cudaEvent_t getEvent() const {
    return event_;
  }

  // Copy contents of device pointer to host.
  void copyToHostAsync(T* dst);

  // Copy contents of host pointer to device.
  void copyFromHostAsync(T* src);

  // Wait for copy to complete.
  void wait();

 protected:
  // Instances must be created through static functions
  CudaDevicePointer(T* ptr, size_t count);

  // Instances cannot be copied or copy-assigned
  CudaDevicePointer(const CudaDevicePointer&) = delete;
  CudaDevicePointer& operator=(const CudaDevicePointer&) = delete;

  // Device pointer
  T* device_;

  // Number of T elements in device pointer
  const size_t count_;

  // GPU that the device pointer lives on
  const int deviceId_;

  // Operations on this pointer are run always run on a stream such
  // that they don't block other operations being executed on the GPU
  // this pointer lives on. The stream can be specified at
  // construction time if one has already been created outside this
  // library. If it is not specified, a new stream is created.
  cudaStream_t stream_ = kStreamNotSet;
  cudaEvent_t event_ = 0;

  // If no stream is specified at construction time, this class
  // allocates a new stream for operations against this pointer.
  // Record whether or not this instance is a stream's owner so that
  // it is destroyed when this instance is destructed.
  bool streamOwner_ = false;
};

} // namespace gloo
