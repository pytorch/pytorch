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

CudaStream::CudaStream(int deviceId, cudaStream_t stream)
    : deviceId_(deviceId),
      stream_(stream),
      streamOwner_(false) {
  CudaDeviceScope scope(deviceId_);

  // Create new stream if it wasn't specified
  if (stream_ == kStreamNotSet) {
    int loPri, hiPri;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&loPri, &hiPri));
    CUDA_CHECK(cudaStreamCreateWithPriority(
                 &stream_, cudaStreamNonBlocking, hiPri));
    streamOwner_ = true;
  }

  // Create new event to synchronize operations against
  CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : deviceId_(other.deviceId_),
      stream_(other.stream_),
      streamOwner_(other.streamOwner_),
      event_(other.event_) {
  other.stream_ = nullptr;
  other.event_ = nullptr;
}

CudaStream::~CudaStream() {
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

template <typename T>
void CudaStream::copyAsync(
    CudaHostPointer<T>& dst,
    CudaDevicePointer<T>& src) {
  CudaDeviceScope scope(deviceId_);
  GLOO_ENFORCE_LE(dst.getCount(), src.getCount());
  CUDA_CHECK(cudaMemcpyAsync(
                 *dst,
                 *src,
                 dst.getCount() * sizeof(T),
                 cudaMemcpyDeviceToHost,
                 stream_));
  CUDA_CHECK(cudaEventRecord(event_, stream_));
}

template <typename T>
void CudaStream::copyAsync(
    CudaHostPointer<T>& dst,
    CudaHostPointer<T>& src) {
  CudaDeviceScope scope(deviceId_);
  GLOO_ENFORCE_LE(dst.getCount(), src.getCount());
  CUDA_CHECK(cudaMemcpyAsync(
                 *dst,
                 *src,
                 dst.getCount() * sizeof(T),
                 cudaMemcpyHostToHost,
                 stream_));
  CUDA_CHECK(cudaEventRecord(event_, stream_));
}

template <typename T>
void CudaStream::copyAsync(
    CudaDevicePointer<T>& dst,
    CudaDevicePointer<T>& src) {
  CudaDeviceScope scope(deviceId_);
  GLOO_ENFORCE_LE(dst.getCount(), src.getCount());
  CUDA_CHECK(cudaMemcpyAsync(
                 *dst,
                 *src,
                 dst.getCount() * sizeof(T),
                 cudaMemcpyDeviceToDevice,
                 stream_));
  CUDA_CHECK(cudaEventRecord(event_, stream_));
}

template <typename T>
void CudaStream::copyAsync(
    CudaDevicePointer<T>& dst,
    CudaHostPointer<T>& src) {
  CudaDeviceScope scope(deviceId_);
  GLOO_ENFORCE_LE(dst.getCount(), src.getCount());
  CUDA_CHECK(cudaMemcpyAsync(
                 *dst,
                 *src,
                 dst.getCount() * sizeof(T),
                 cudaMemcpyHostToDevice,
                 stream_));
  CUDA_CHECK(cudaEventRecord(event_, stream_));
}

void CudaStream::record() {
  CUDA_CHECK(cudaEventRecord(event_, stream_));
}

void CudaStream::wait() {
  CudaDeviceScope scope(deviceId_);
  CUDA_CHECK(cudaEventSynchronize(event_));
}

template <typename T>
CudaDevicePointer<T> CudaDevicePointer<T>::alloc(
    size_t count) {
  T* ptr = nullptr;
  size_t bytes = count * sizeof(T);
  {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
  }
  auto p = create(ptr, count);
  p.owner_ = true;
  return p;
}

template<typename T>
CudaDevicePointer<T> CudaDevicePointer<T>::create(
    T* ptr,
    size_t count) {
  CudaDevicePointer p(ptr, count, false);
  return p;
}

template<typename T>
CudaDevicePointer<T>::CudaDevicePointer(T* ptr, size_t count, bool owner)
    : device_(ptr),
      count_(count),
      owner_(owner),
      deviceId_(getGPUIDForPointer(device_)) {
}

template<typename T>
CudaDevicePointer<T>::CudaDevicePointer(CudaDevicePointer<T>&& other) noexcept
    : device_(other.device_),
      count_(other.count_),
      owner_(other.owner_),
      deviceId_(other.deviceId_) {
  // Nullify fields that would otherwise be destructed
  other.device_ = nullptr;
  other.owner_ = false;
}

template<typename T>
CudaDevicePointer<T>& CudaDevicePointer<T>::operator=(
    CudaDevicePointer<T>&& other) {
  device_ = other.device_;
  count_ = other.count_;
  owner_ = other.owner_;
  deviceId_ = other.deviceId_;

  // Nullify fields that would otherwise be destructed
  other.device_ = nullptr;
  other.owner_ = false;

  return *this;
}

template<typename T>
CudaDevicePointer<T>::~CudaDevicePointer() {
  if (deviceId_ < 0) {
    return;
  }
  CudaDeviceScope scope(deviceId_);
  if (owner_ && device_ != nullptr) {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    CUDA_CHECK(cudaFree(device_));
  }
}

template <typename T>
CudaHostPointer<T> CudaHostPointer<T>::alloc(size_t count) {
  T* ptr = nullptr;
  size_t bytes = count * sizeof(T);
  {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    CUDA_CHECK(cudaMallocHost(&ptr, bytes));
  }
  return CudaHostPointer<T>(ptr, count, true);
}

template <typename T>
CudaHostPointer<T>::CudaHostPointer(T* ptr, size_t count, bool owner)
    : host_(ptr),
      count_(count),
      owner_(owner) {}

template <typename T>
CudaHostPointer<T>::CudaHostPointer(CudaHostPointer&& other) noexcept
    : host_(other.host_),
      count_(other.count_),
      owner_(other.owner_) {
  other.host_ = nullptr;
  other.count_ = 0;
  other.owner_ = false;
}

template<typename T>
CudaHostPointer<T>& CudaHostPointer<T>::operator=(CudaHostPointer<T>&& other) {
  host_ = other.host_;
  count_ = other.count_;
  owner_ = other.owner_;
  other.host_ = nullptr;
  other.count_ = 0;
  other.owner_ = false;
  return *this;
}

template<typename T>
CudaHostPointer<T>::~CudaHostPointer() {
  if (owner_) {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    CUDA_CHECK(cudaFreeHost(host_));
  }
}

// Instantiate templates
template class CudaDevicePointer<float>;
template class CudaHostPointer<float>;

template void CudaStream::copyAsync<float>(
    CudaHostPointer<float>& dst,
    CudaDevicePointer<float>& src);

template void CudaStream::copyAsync<float>(
    CudaHostPointer<float>& dst,
    CudaHostPointer<float>& src);

template void CudaStream::copyAsync<float>(
    CudaDevicePointer<float>& dst,
    CudaDevicePointer<float>& src);

template void CudaStream::copyAsync<float>(
    CudaDevicePointer<float>& dst,
    CudaHostPointer<float>& src);

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
