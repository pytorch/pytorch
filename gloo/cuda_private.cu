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

template<typename T>
__global__ void initializeMemory(T* ptr, const T val, const size_t n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += blockDim.x) {
    ptr[i] = val;
  }
}

template<typename T>
CudaMemory<T>::CudaMemory(size_t n, T val): n_(n), bytes_(n * sizeof(T)) {
  CUDA_CHECK(cudaGetDevice(&device_));
  CUDA_CHECK(cudaMalloc(&ptr_, bytes_));
  initializeMemory<<<1, 32>>>(ptr_, val, n);
}

template<typename T>
CudaMemory<T>::~CudaMemory() {
  CudaDeviceGuard guard;
  CUDA_CHECK(cudaSetDevice(device_));
  CUDA_CHECK(cudaFree(ptr_));
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
