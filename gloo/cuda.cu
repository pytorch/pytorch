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

template <typename T>
CudaDevicePointer<T> CudaDevicePointer<T>::alloc(
    size_t count,
    cudaStream_t stream) {
  T* ptr = nullptr;
  size_t bytes = count * sizeof(T);
  CUDA_CHECK(cudaMalloc(&ptr, bytes));
  auto p = create(ptr, count, stream);
  p.owner_ = true;
  return p;
}

template<typename T>
CudaDevicePointer<T> CudaDevicePointer<T>::create(
    T* ptr,
    size_t count,
    cudaStream_t stream) {
  CudaDevicePointer p(ptr, count, false);

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
CudaDevicePointer<T>::CudaDevicePointer(T* ptr, size_t count, bool owner)
    : device_(ptr),
      count_(count),
      owner_(owner),
      deviceId_(getGPUIDForPointer(device_)) {
  CudaDeviceScope scope(deviceId_);
  CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
}

template<typename T>
CudaDevicePointer<T>::CudaDevicePointer(CudaDevicePointer<T>&& other) noexcept
    : device_(other.device_),
      count_(other.count_),
      owner_(other.owner_),
      deviceId_(other.deviceId_),
      streamOwner_(other.streamOwner_),
      stream_(other.stream_),
      event_(other.event_) {
  // Nullify fields that would otherwise be destructed
  other.device_ = nullptr;
  other.owner_ = false;
  other.stream_ = nullptr;
  other.event_ = nullptr;
}

template<typename T>
CudaDevicePointer<T>& CudaDevicePointer<T>::operator=(
    CudaDevicePointer<T>&& other) {
  device_ = other.device_;
  count_ = other.count_;
  owner_ = other.owner_;
  deviceId_ = other.deviceId_;
  streamOwner_ = other.streamOwner_;
  stream_ = other.stream_;
  event_ = other.event_;

  // Nullify fields that would otherwise be destructed
  other.device_ = nullptr;
  other.owner_ = false;
  other.stream_ = nullptr;
  other.event_ = nullptr;

  return *this;
}

template<typename T>
CudaDevicePointer<T>::~CudaDevicePointer() {
  CudaDeviceScope scope(deviceId_);
  if (owner_ && device_ != nullptr) {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    CUDA_CHECK(cudaFree(device_));
  }
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
void CudaDevicePointer<T>::copyToAsync(CudaHostPointer<T>& dst) {
  CudaDeviceScope scope(deviceId_);
  CUDA_CHECK(cudaMemcpyAsync(
               *dst,
               device_,
               count_ * sizeof(T),
               cudaMemcpyDeviceToHost,
               stream_));
  CUDA_CHECK(cudaEventRecord(event_, stream_));
}

template<typename T>
void CudaDevicePointer<T>::copyFromAsync(CudaHostPointer<T>& src) {
  CudaDeviceScope scope(deviceId_);
  CUDA_CHECK(cudaMemcpyAsync(
               device_,
               *src,
               count_ * sizeof(T),
               cudaMemcpyHostToDevice,
               stream_));
  CUDA_CHECK(cudaEventRecord(event_, stream_));
}

template<typename T>
void CudaDevicePointer<T>::copyToAsync(CudaDevicePointer<T>& dst) {
  CudaDeviceScope scope(deviceId_);
  CUDA_CHECK(cudaMemcpyAsync(
               *dst,
               device_,
               count_ * sizeof(T),
               cudaMemcpyDeviceToDevice,
               stream_));
  CUDA_CHECK(cudaEventRecord(event_, stream_));
}

template<typename T>
void CudaDevicePointer<T>::copyFromAsync(CudaDevicePointer<T>& src) {
  CudaDeviceScope scope(deviceId_);
  CUDA_CHECK(cudaMemcpyAsync(
               device_,
               *src,
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

template <typename T>
CudaHostPointer<T> CudaHostPointer<T>::alloc(size_t count) {
  T* ptr = nullptr;
  size_t bytes = count * sizeof(T);
  CUDA_CHECK(cudaMallocHost(&ptr, bytes));
  return CudaHostPointer<T>(ptr, count, true);
}

template <typename T>
CudaHostPointer<T>::CudaHostPointer(T* ptr, size_t count, bool owner)
    : host_(ptr),
      count_(count),
      owner_(owner),
      deviceId_(-1),
      stream_(nullptr),
      event_(nullptr) {}

template <typename T>
CudaHostPointer<T>::CudaHostPointer(CudaHostPointer&& other) noexcept
    : host_(other.host_),
      count_(other.count_),
      owner_(other.owner_),
      deviceId_(other.deviceId_),
      stream_(other.stream_),
      event_(other.event_) {
  other.host_ = nullptr;
  other.count_ = 0;
  other.owner_ = false;
  other.deviceId_ = -1;
  other.stream_ = nullptr;
  other.event_ = nullptr;
}

template<typename T>
CudaHostPointer<T>& CudaHostPointer<T>::operator=(CudaHostPointer<T>&& other) {
  host_ = other.host_;
  count_ = other.count_;
  owner_ = other.owner_;
  deviceId_ = other.deviceId_;
  stream_ = other.stream_;
  event_ = other.event_;
  other.host_ = nullptr;
  other.count_ = 0;
  other.owner_ = false;
  other.deviceId_ = -1;
  other.stream_ = nullptr;
  other.event_ = nullptr;
  return *this;
}

template<typename T>
CudaHostPointer<T>::~CudaHostPointer() {
  if (owner_) {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    CUDA_CHECK(cudaFreeHost(host_));
  }
}

template<typename T>
void CudaHostPointer<T>::copyToAsync(CudaHostPointer<T>& dst) {
  wait();
  memcpy(dst.host_, host_, count_ * sizeof(T));
}

template<typename T>
void CudaHostPointer<T>::copyFromAsync(CudaHostPointer<T>& src) {
  wait();
  memcpy(host_, src.host_, count_ *  sizeof(T));
}

template<typename T>
void CudaHostPointer<T>::copyToAsync(CudaDevicePointer<T>& dst) {
  // Wait for completion of in-flight copy if destination uses different stream.
  if (stream_ != nullptr && dst.getStream() != stream_) {
    wait();
  }

  dst.copyFromAsync(*this);
  dst.wait();
  deviceId_ = dst.getDeviceID();
  stream_ = dst.getStream();
  event_ = dst.getEvent();
}

template<typename T>
void CudaHostPointer<T>::copyFromAsync(CudaDevicePointer<T>& src) {
  // Wait for completion of in-flight copy if source uses different stream.
  if (stream_ != nullptr && src.getStream() != stream_) {
    wait();
  }

  src.copyToAsync(*this);
  src.wait();
  deviceId_ = src.getDeviceID();
  stream_ = src.getStream();
  event_ = src.getEvent();
}

template<typename T>
void CudaHostPointer<T>::wait() {
  // If a copy operation stored the corresponding CUDA event we
  // wait for it. Otherwise there is nothing to wait for.
  if (event_ != nullptr) {
    CudaDeviceScope scope(deviceId_);
    CUDA_CHECK(cudaEventSynchronize(event_));
    deviceId_ = -1;
    stream_ = nullptr;
    event_ = nullptr;
  }
}

// Instantiate templates
template class CudaDevicePointer<float>;
template class CudaHostPointer<float>;

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
