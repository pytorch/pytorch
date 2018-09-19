#include "ATen/cuda/CUDAContext.h"
#include "THC/THCGeneral.hpp"

namespace at { namespace cuda { 

/* Device info */
int64_t getNumGPUs() {
  int count;
  AT_CUDA_CHECK(cudaGetDeviceCount(&count));
  return count;
}

int64_t current_device() {
  int cur_device;
  AT_CUDA_CHECK(cudaGetDevice(&cur_device));
  return cur_device;
}

void set_device(int64_t device) {
  AT_CUDA_CHECK(cudaSetDevice((int)device));
}

cudaDeviceProp* getCurrentDeviceProperties() {
  return THCState_getCurrentDeviceProperties(at::globalContext().getTHCState());
}

cudaDeviceProp* getDeviceProperties(int64_t device) {
  return THCState_getDeviceProperties(at::globalContext().getTHCState(), (int)device);
}

/* Streams */
CUDAStream createCUDAStream(
  const bool isHighPriority
, int64_t device) {
  return detail::CUDAStream_createStream(isHighPriority, device);
}

CUDAStream getDefaultCUDAStream(int64_t device) {
  return detail::CUDAStream_getDefaultStream(device);
}
CUDAStream getCurrentCUDAStream(int64_t device) {
  return detail::CUDAStream_getCurrentStream(device);
}

void setCurrentCUDAStream(CUDAStream stream) {
  detail::CUDAStream_setStream(stream.internals());
}
void uncheckedSetCurrentCUDAStream(CUDAStream stream) {
  detail::CUDAStream_uncheckedSetStream(stream.internals());
}

Allocator* getCUDADeviceAllocator() {
  return at::globalContext().getTHCState()->cudaDeviceAllocator;
}

/* Handles */
#ifndef __HIP_PLATFORM_HCC__
  cusparseHandle_t getCurrentCUDASparseHandle() {
    return THCState_getCurrentSparseHandle(at::globalContext().getTHCState());
  }
#endif

} // namespace cuda

} // namespace at
