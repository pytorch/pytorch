#pragma once

#include "ATen/ATenGeneral.h"
#include "ATen/Context.h"
#include "ATen/cuda/CUDAStream.h"
#include "ATen/cuda/Exceptions.h"

#include <cstdint>

#include "cuda_runtime_api.h"
#include "cusparse.h"

namespace at {
namespace cuda {

/*
A common CUDA interface for ATen.

This interface is distinct from CUDAHooks, which defines an interface that links
to both CPU-only and CUDA builds. That interface is intended for runtime
dispatch and should be used from files that are included in both CPU-only and
CUDA builds.

CUDAContext, on the other hand, should be preferred by files only included in
CUDA builds. It is intended to expose CUDA functionality in a consistent
manner.

This means there is some overlap between the CUDAContext and CUDAHooks, but
the choice of which to use is simple: use CUDAContext when in a CUDA-only file,
use CUDAHooks otherwise.

Note that CUDAContext simply defines an interface with no associated class.
It is expected that the modules whose functions compose this interface will
manage their own state. There is only a single CUDA context/state.
*/

/* Device info */
AT_API int64_t getNumGPUs();

AT_API int64_t current_device();

AT_API cudaDeviceProp* getCurrentDeviceProperties();

AT_API cudaDeviceProp* getDeviceProperties(int64_t device);

/* Streams */
AT_API CUDAStream createCUDAStream();

AT_API CUDAStream createCUDAStreamWithOptions(int32_t flags, int32_t priority);

AT_API CUDAStream getDefaultCUDAStream();

AT_API CUDAStream getDefaultCUDAStreamOnDevice(int64_t device);

AT_API CUDAStream getCurrentCUDAStream();

AT_API CUDAStream getCurrentCUDAStreamOnDevice(int64_t device);

AT_API void setCurrentCUDAStream(CUDAStream stream);

AT_API void setCurrentCUDAStreamOnDevice(int64_t device, CUDAStream stream);

AT_API void uncheckedSetCurrentCUDAStreamOnDevice(int64_t device, CUDAStream stream);

/* Handles */
#ifndef __HIP_PLATFORM_HCC__
  AT_API cusparseHandle_t getCurrentCUDASparseHandle();
#endif


} // namespace cuda
} // namespace at
