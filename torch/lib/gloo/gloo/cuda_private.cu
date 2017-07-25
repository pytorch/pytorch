/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_private.h"

#include <cuda_fp16.h>

#include "gloo/common/common.h"
#include "gloo/types.h"

namespace gloo {

template<typename T>
__global__ void initializeMemory(
    T* ptr,
    const int val,
    const size_t count,
    const size_t stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < count; i += blockDim.x) {
    ptr[i] = (i * stride) + val;
  }
}

template<>
__global__ void initializeMemory<float16>(
    float16* ptr,
    const int val,
    const size_t count,
    const size_t stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  half* ptrAsHalf = (half*) ptr;
  for (; i < count; i += blockDim.x) {
    ptrAsHalf[i] = __float2half(static_cast<float>((i * stride) + val));
  }
}

template<typename T>
CudaMemory<T>::CudaMemory(size_t elements)
    : elements(elements),
      bytes(elements * sizeof(T)) {
  CUDA_CHECK(cudaGetDevice(&device_));
  // Sychronize memory allocation with NCCL operations
  std::lock_guard<std::mutex> lock(CudaShared::getMutex());
  CUDA_CHECK(cudaMalloc(&ptr_, bytes));
}

template<typename T>
CudaMemory<T>::CudaMemory(CudaMemory<T>&& other) noexcept
  : elements(other.elements),
    bytes(other.bytes),
    device_(other.device_),
    ptr_(other.ptr_) {
  // Nullify pointer on move source
  other.ptr_ = nullptr;
}

template<typename T>
CudaMemory<T>::~CudaMemory() {
  CudaDeviceScope scope(device_);
  if (ptr_ != nullptr) {
    // Sychronize memory allocation with NCCL operations
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    CUDA_CHECK(cudaFree(ptr_));
  }
}

template<typename T>
void CudaMemory<T>::set(int val, size_t stride, cudaStream_t stream) {
  CudaDeviceScope scope(device_);
  if (stream == kStreamNotSet) {
    initializeMemory<T><<<1, 32>>>(ptr_, val, elements, stride);
  } else {
    initializeMemory<T><<<1, 32, 0, stream>>>(ptr_, val, elements, stride);
  }
}

template<typename T>
std::unique_ptr<T[]> CudaMemory<T>::copyToHost() const {
  auto host = make_unique<T[]>(elements);
  cudaMemcpy(host.get(), ptr_, bytes, cudaMemcpyDefault);
  return host;
}

// Instantiate template
template class CudaMemory<float>;
template class CudaMemory<float16>;

} // namespace gloo
