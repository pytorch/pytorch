// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <cstdint>

namespace torch::comms {

__global__ void atomicAddKernel(uint64_t* counter, uint64_t amount) {
  atomicAdd(reinterpret_cast<unsigned long long*>(counter), amount);
}

__attribute__((weak)) cudaError_t
launchAtomicAdd(cudaStream_t stream, uint64_t* d_counter, uint64_t amount) {
  atomicAddKernel<<<1, 1, 0, stream>>>(d_counter, amount);
  return cudaGetLastError();
}

} // namespace torch::comms
