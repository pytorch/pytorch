#include "ATen/cuda/CUDAContext.h"
#include "THC/THCGeneral.h"

namespace at { namespace cuda { 

/* Device info */
cudaDeviceProp* getCurrentDeviceProperties() {
  return THCState_getCurrentDeviceProperties(at::globalContext().getTHCState());
}

cudaDeviceProp* getDeviceProperties(int64_t device) {
  return THCState_getDeviceProperties(at::globalContext().getTHCState(), (int)device);
}

/* Handles */
#ifndef __HIP_PLATFORM_HCC__
  cusparseHandle_t getCurrentCUDASparseHandle() {
    return THCState_getCurrentSparseHandle(at::globalContext().getTHCState());
  }
#endif

} // namespace cuda

} // namespace at