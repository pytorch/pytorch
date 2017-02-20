/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda.h"
#include "gloo/cuda_private.h"

namespace gloo {

const cudaStream_t kStreamNotSet = (cudaStream_t)(-1);

template<typename T>
CudaDevicePointer<T>
CudaDevicePointer<T>::create(
    T* ptr,
    size_t count,
    cudaStream_t stream) {
  CudaDevicePointer p(ptr, count);

  // Create new stream for operations concerning this pointer
  if (stream == kStreamNotSet) {
    CudaDeviceScope scope(p.deviceId_);
    int loPri, hiPri;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&loPri, &hiPri));
    CUDA_CHECK(cudaStreamCreateWithPriority(
                 &p.stream_, cudaStreamNonBlocking, hiPri));
    p.streamOwner_ = true;
  } else {
    p.stream_ = stream;
  }

  return p;
}

template<typename T>
CudaDevicePointer<T>::CudaDevicePointer(T* ptr, size_t count)
    : device_(ptr),
      count_(count),
      deviceId_(getGPUIDForPointer(device_)) {
  CudaDeviceScope scope(deviceId_);
  CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
}

template<typename T>
CudaDevicePointer<T>::CudaDevicePointer(CudaDevicePointer<T>&& other) noexcept
    : device_(other.device_),
      count_(other.count_),
      deviceId_(other.deviceId_),
      streamOwner_(other.streamOwner_),
      stream_(other.stream_),
      event_(other.event_) {
  // Nullify fields that would otherwise be destructed
  other.stream_ = nullptr;
  other.event_ = nullptr;
}

template<typename T>
CudaDevicePointer<T>::~CudaDevicePointer() {
  CudaDeviceScope scope(deviceId_);
  if (streamOwner_ && stream_ != nullptr) {
    CUDA_CHECK(cudaStreamDestroy(stream_));
  }
  if (event_ != nullptr) {
    CUDA_CHECK(cudaEventDestroy(event_));
  }
}

template<typename T>
void CudaDevicePointer<T>::copyToHostAsync(T* dst) {
  CudaDeviceScope scope(deviceId_);
  CUDA_CHECK(cudaMemcpyAsync(
               dst,
               device_,
               count_ * sizeof(T),
               cudaMemcpyDeviceToHost,
               stream_));
  CUDA_CHECK(cudaEventRecord(event_, stream_));
}

template<typename T>
void CudaDevicePointer<T>::copyFromHostAsync(T* src) {
  CudaDeviceScope scope(deviceId_);
  CUDA_CHECK(cudaMemcpyAsync(
               device_,
               src,
               count_ * sizeof(T),
               cudaMemcpyDeviceToHost,
               stream_));
  CUDA_CHECK(cudaEventRecord(event_, stream_));
}

template<typename T>
void CudaDevicePointer<T>::waitAsync() {
  CudaDeviceScope scope(deviceId_);
  CUDA_CHECK(cudaEventSynchronize(event_));
}

// Instantiate template
template class CudaDevicePointer<float>;

} // namespace gloo
