#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>

#ifdef CUDART_VERSION
#include <cusolverDn.h>
#endif

#include <ATen/core/ATenGeneral.h>
#include <ATen/Context.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Logging.h>
#include <ATen/cuda/Exceptions.h>

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

/**
 * DEPRECATED: use device_count() instead
 */
inline int64_t getNumGPUs() {
    return c10::cuda::device_count();
}

/**
 * CUDA is available if we compiled with CUDA, and there are one or more
 * devices.  If we compiled with CUDA but there is a driver problem, etc.,
 * this function will report CUDA is not available (rather than raise an error.)
 */
inline bool is_available() {
    return c10::cuda::device_count() > 0;
}

TORCH_CUDA_CPP_API cudaDeviceProp* getCurrentDeviceProperties();

TORCH_CUDA_CPP_API int warp_size();

TORCH_CUDA_CPP_API cudaDeviceProp* getDeviceProperties(int64_t device);

TORCH_CUDA_CPP_API bool canDeviceAccessPeer(
    int64_t device,
    int64_t peer_device);

TORCH_CUDA_CPP_API Allocator* getCUDADeviceAllocator();

/* Handles */
TORCH_CUDA_CPP_API cusparseHandle_t getCurrentCUDASparseHandle();
TORCH_CUDA_CPP_API cublasHandle_t getCurrentCUDABlasHandle();

TORCH_CUDA_CPP_API void clearCublasWorkspaces();

#ifdef CUDART_VERSION
TORCH_CUDA_CPP_API cusolverDnHandle_t getCurrentCUDASolverDnHandle();
#endif

} // namespace cuda
} // namespace at
