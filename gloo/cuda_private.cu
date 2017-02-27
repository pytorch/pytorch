/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_private.h"

#include "gloo/common/common.h"

namespace gloo {

std::mutex gCudaMutex;

template<typename T>
__global__ void initializeMemory(T* ptr, const T val, const size_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += blockDim.x) {
    ptr[i] = val;
  }
}

template<typename T>
CudaMemory<T>::CudaMemory(size_t n): n_(n), bytes_(n * sizeof(T)) {
  CUDA_CHECK(cudaGetDevice(&device_));
  // Sychronize memory allocation with NCCL operations
  std::lock_guard<std::mutex> lock(gCudaMutex);
  CUDA_CHECK(cudaMalloc(&ptr_, bytes_));
}

template<typename T>
CudaMemory<T>::CudaMemory(CudaMemory<T>&& other) noexcept
  : n_(other.n_),
    bytes_(other.bytes_),
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
    std::lock_guard<std::mutex> lock(gCudaMutex);
    CUDA_CHECK(cudaFree(ptr_));
  }
}

template<typename T>
void CudaMemory<T>::set(T val, cudaStream_t stream) {
  CudaDeviceScope scope(device_);
  if (stream == kStreamNotSet) {
    initializeMemory<<<1, 32>>>(ptr_, val, n_);
  } else {
    initializeMemory<<<1, 32, 0, stream>>>(ptr_, val, n_);
  }
}

template<typename T>
std::unique_ptr<T[]> CudaMemory<T>::copyToHost() {
  auto host = make_unique<T[]>(n_);
  cudaMemcpy(host.get(), ptr_, bytes_, cudaMemcpyDefault);
  return host;
}

// Instantiate template
template class CudaMemory<float>;

} // namespace gloo
