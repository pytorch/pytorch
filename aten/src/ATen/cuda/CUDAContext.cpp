#include "ATen/cuda/CUDAContext.h"
#include "THC/THCGeneral.h"

namespace at { namespace cuda { 

/* Device info */
AT_API cudaDeviceProp* getCurrentDeviceProperties() {
  return THCState_getCurrentDeviceProperties(at::globalContext().getTHCState());
}

AT_API cudaDeviceProp* getDeviceProperties(int64_t device) {
  return THCState_getDeviceProperties(at::globalContext().getTHCState(), (int)device);
}

/* Handles */
#ifndef __HIP_PLATFORM_HCC__
  AT_API cusparseHandle_t getCurrentCUDASparseHandle() {
    return THCState_getCurrentSparseHandle(at::globalContext().getTHCState());
  }
#endif

} // namespace cuda

} // namespace at