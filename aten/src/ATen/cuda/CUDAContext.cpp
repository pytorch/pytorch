#include "ATen/cuda/CUDAContext.h"
#include "THC/THCGeneral.h"

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

cudaDeviceProp* getCurrentDeviceProperties() {
  return THCState_getCurrentDeviceProperties(at::globalContext().getTHCState());
}

cudaDeviceProp* getDeviceProperties(int64_t device) {
  return THCState_getDeviceProperties(at::globalContext().getTHCState(), (int)device);
}

/* Streams */
CUDAStream createCUDAStream() {
  return detail::CUDAStream_createAndRetainWithOptions(
    CUDAStream::DEFAULT_FLAGS
  , CUDAStream::DEFAULT_PRIORITY
  );
}

CUDAStream createCUDAStreamWithOptions(int32_t flags, int32_t priority) {
  return detail::CUDAStream_createAndRetainWithOptions(flags, priority);
}

CUDAStream getDefaultCUDAStream() {
  return detail::CUDAStream_getDefaultStream();
}

CUDAStream getDefaultCUDAStreamOnDevice(int64_t device) {
  return detail::CUDAStream_getDefaultStreamOnDevice(device);
}

CUDAStream getCurrentCUDAStream() {
  return detail::CUDAStream_getAndRetainCurrentStream();
}

CUDAStream getCurrentCUDAStreamOnDevice(int64_t device) {
  return detail::CUDAStream_getAndRetainCurrentStreamOnDevice(device);
}

void setCurrentCUDAStream(CUDAStream stream) {
  return detail::CUDAStream_setStream(stream.internals());
}

void setCurrentCUDAStreamOnDevice(int64_t device, CUDAStream stream) {
  return detail::CUDAStream_setStreamOnDevice(device, stream.internals());
}

void uncheckedSetCurrentCUDAStreamOnDevice(int64_t device, CUDAStream stream) {
  return detail::CUDAStream_uncheckedSetStreamOnDevice(device, stream.internals());
}

/* Handles */
#ifndef __HIP_PLATFORM_HCC__
  cusparseHandle_t getCurrentCUDASparseHandle() {
    return THCState_getCurrentSparseHandle(at::globalContext().getTHCState());
  }
#endif

} // namespace cuda

} // namespace at