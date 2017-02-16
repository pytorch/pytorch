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

namespace gloo {

template<typename T>
class CudaDevicePointer {
 public:
  static CudaDevicePointer createWithNewStream(
    T* ptr,
    size_t count);

  static CudaDevicePointer createWithStream(
    T* ptr,
    size_t count,
    cudaStream_t stream);

  CudaDevicePointer(CudaDevicePointer&&) noexcept;
  ~CudaDevicePointer();

  // Copy contents of device pointer to host.
  void copyToHostAsync(T* dst);

  // Copy contents of host pointer to device.
  void copyFromHostAsync(T* src);

  // Wait for copy to complete.
  void waitAsync();

 protected:
  // Instances must be created through static functions
  CudaDevicePointer(T* ptr, size_t count);

  // Instances cannot be copied or assigned
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
  cudaStream_t stream_ = 0;
  cudaEvent_t event_ = 0;

  // If no stream is specified at construction time, this class
  // allocates a new stream for operations against this pointer.
  // Record whether or not this instance is a stream's owner so that
  // it is destroyed when this instance is destructed.
  bool streamOwner_ = false;
};

} // namespace gloo
