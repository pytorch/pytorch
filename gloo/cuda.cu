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

// Default mutex to synchronize contentious CUDA and NCCL operations
static std::mutex defaultCudaMutex;
std::atomic<std::mutex*> CudaShared::mutex_(&defaultCudaMutex);

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
CudaDevicePointer<T>& CudaDevicePointer<T>::operator=(
    CudaDevicePointer<T>&& other) {
  device_ = other.device_;
  count_ = other.count_;
  deviceId_ = other.deviceId_;
  streamOwner_ = other.streamOwner_;
  stream_ = other.stream_;
  event_ = other.event_;

  // Nullify fields that would otherwise be destructed
  other.stream_ = nullptr;
  other.event_ = nullptr;

  return *this;
}

template<typename T>
CudaDevicePointer<T>::~CudaDevicePointer() {
  CudaDeviceScope scope(deviceId_);
  if (event_ != nullptr) {
    // Make sure outstanding operations are complete. If the event
    // hasn't been queued this call will return immediately.
    CUDA_CHECK(cudaEventSynchronize(event_));
    CUDA_CHECK(cudaEventDestroy(event_));
  }
  if (streamOwner_ && stream_ != nullptr) {
    CUDA_CHECK(cudaStreamDestroy(stream_));
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
void CudaDevicePointer<T>::copyToDeviceAsync(T* dst) {
  CudaDeviceScope scope(deviceId_);
  CUDA_CHECK(cudaMemcpyAsync(
               dst,
               device_,
               count_ * sizeof(T),
               cudaMemcpyDeviceToDevice,
               stream_));
  CUDA_CHECK(cudaEventRecord(event_, stream_));
}

template<typename T>
void CudaDevicePointer<T>::copyFromDeviceAsync(T* src) {
  CudaDeviceScope scope(deviceId_);
  CUDA_CHECK(cudaMemcpyAsync(
               device_,
               src,
               count_ * sizeof(T),
               cudaMemcpyDeviceToDevice,
               stream_));
  CUDA_CHECK(cudaEventRecord(event_, stream_));
}

template<typename T>
void CudaDevicePointer<T>::wait() {
  CudaDeviceScope scope(deviceId_);
  CUDA_CHECK(cudaEventSynchronize(event_));
}

// Instantiate template
template class CudaDevicePointer<float>;

// Borrowed limits from Caffe2 code (see core/common_gpu.h)
constexpr static int kCudaNumThreads = 512;
constexpr static int kCudaMaximumNumBlocks = 4096;

static inline int cudaGetBlocks(const int N) {
  return std::min((N + kCudaNumThreads - 1) / kCudaNumThreads,
                  kCudaMaximumNumBlocks);
}

#define DELEGATE_SIMPLE_CUDA_BINARY_OPERATOR(T, Funcname, op)           \
  __global__                                                            \
  void _Kernel_##T##_##Funcname(T* dst, const T* src, const int n) {    \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;                 \
         i < (n);                                                       \
         i += blockDim.x * gridDim.x) {                                 \
      dst[i] = dst[i] op src[i];                                        \
    }                                                                   \
  }                                                                     \
  template <>                                                           \
  void Funcname<T>(                                                     \
    T* dst,                                                             \
    const T* src,                                                       \
    size_t n,                                                           \
    const cudaStream_t stream) {                                        \
    _Kernel_##T##_##Funcname<<<                                         \
      cudaGetBlocks(n),                                                 \
      kCudaNumThreads,                                                  \
      0,                                                                \
      stream>>>(                                                        \
        dst, src, n);                                                   \
  }

DELEGATE_SIMPLE_CUDA_BINARY_OPERATOR(float, cudaSum, +);
DELEGATE_SIMPLE_CUDA_BINARY_OPERATOR(float, cudaProduct, *);

#define DELEGATE_SIMPLE_CUDA_BINARY_COMPARE(T, Funcname, op)            \
  __global__                                                            \
  void _Kernel_##T##_##Funcname(T* dst, const T* src, const int n) {    \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;                 \
         i < (n);                                                       \
         i += blockDim.x * gridDim.x) {                                 \
      if (src[i] op dst[i]) {                                           \
        dst[i] = src[i];                                                \
      }                                                                 \
    }                                                                   \
  }                                                                     \
  template <>                                                           \
  void Funcname<T>(                                                     \
    T* dst,                                                             \
    const T* src,                                                       \
    size_t n,                                                           \
    const cudaStream_t stream) {                                        \
    _Kernel_##T##_##Funcname<<<                                         \
      cudaGetBlocks(n),                                                 \
      kCudaNumThreads,                                                  \
      0,                                                                \
      stream>>>(                                                        \
        dst, src, n);                                                   \
  }

DELEGATE_SIMPLE_CUDA_BINARY_COMPARE(float, cudaMin, <);
DELEGATE_SIMPLE_CUDA_BINARY_COMPARE(float, cudaMax, >);

} // namespace gloo
