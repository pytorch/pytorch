#include <ATen/cuda/CUDAContext.h>
#include <THC/THCGeneral.hpp>

#include <ATen/cuda/CUDAConfig.h>

namespace at { namespace cuda {

/* Device info */
int warp_size() {
  return getCurrentDeviceProperties()->warpSize;
}

cudaDeviceProp* getCurrentDeviceProperties() {
  return THCState_getCurrentDeviceProperties(at::globalContext().getTHCState());
}

cudaDeviceProp* getDeviceProperties(int64_t device) {
  return THCState_getDeviceProperties(at::globalContext().getTHCState(), (int)device);
}

Allocator* getCUDADeviceAllocator() {
  return at::globalContext().getTHCState()->cudaDeviceAllocator;
}

/* Handles */
cusparseHandle_t getCurrentCUDASparseHandle() {
  return THCState_getCurrentSparseHandle(at::globalContext().getTHCState());
}

cublasHandle_t getCurrentCUDABlasHandle() {
  return THCState_getCurrentBlasHandle(at::globalContext().getTHCState());
}

} // namespace cuda

} // namespace at
