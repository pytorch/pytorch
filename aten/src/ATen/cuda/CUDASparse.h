#pragma once

#include <ATen/cuda/CUDAContext.h>
#if defined(USE_ROCM)
#include <hipsparse/hipsparse-version.h>
#define HIPSPARSE_VERSION ((hipsparseVersionMajor*100000) + (hipsparseVersionMinor*100) + hipsparseVersionPatch)
#endif


// cuSparse Generic API spsv function was added in CUDA 11.3.0
#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION)
#define AT_USE_CUSPARSE_GENERIC_SPSV() 1
#else
#define AT_USE_CUSPARSE_GENERIC_SPSV() 0
#endif

// cuSparse Generic API spsm function was added in CUDA 11.3.1
#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION)
#define AT_USE_CUSPARSE_GENERIC_SPSM() 1
#else
#define AT_USE_CUSPARSE_GENERIC_SPSM() 0
#endif
