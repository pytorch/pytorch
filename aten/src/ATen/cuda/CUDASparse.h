#pragma once

#include <ATen/cuda/CUDAContext.h>
#if defined(USE_ROCM)
#include <hipsparse/hipsparse-version.h>
#define HIPSPARSE_VERSION ((hipsparseVersionMajor*100000) + (hipsparseVersionMinor*100) + hipsparseVersionPatch)
#endif

// cuSparse Generic API added in CUDA 10.1
// Windows support added in CUDA 11.0
#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && ((CUSPARSE_VERSION >= 10300) || (CUSPARSE_VERSION >= 11000 && defined(_WIN32)))
#define AT_USE_CUSPARSE_GENERIC_API() 1
#else
#define AT_USE_CUSPARSE_GENERIC_API() 0
#endif

// cuSparse Generic API descriptor pointers were changed to const in CUDA 12.0
#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && \
    (CUSPARSE_VERSION < 12000)
#define AT_USE_CUSPARSE_NON_CONST_DESCRIPTORS() 1
#else
#define AT_USE_CUSPARSE_NON_CONST_DESCRIPTORS() 0
#endif

#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && \
    (CUSPARSE_VERSION >= 12000)
#define AT_USE_CUSPARSE_CONST_DESCRIPTORS() 1
#else
#define AT_USE_CUSPARSE_CONST_DESCRIPTORS() 0
#endif

#if defined(USE_ROCM)
// hipSparse const API added in v2.4.0
#if HIPSPARSE_VERSION >= 200400
#define AT_USE_HIPSPARSE_CONST_DESCRIPTORS() 1
#define AT_USE_HIPSPARSE_NON_CONST_DESCRIPTORS() 0
#define AT_USE_HIPSPARSE_GENERIC_API() 1
#else
#define AT_USE_HIPSPARSE_CONST_DESCRIPTORS() 0
#define AT_USE_HIPSPARSE_NON_CONST_DESCRIPTORS() 1
#define AT_USE_HIPSPARSE_GENERIC_API() 1
#endif
#else // USE_ROCM
#define AT_USE_HIPSPARSE_CONST_DESCRIPTORS() 0
#define AT_USE_HIPSPARSE_NON_CONST_DESCRIPTORS() 0
#define AT_USE_HIPSPARSE_GENERIC_API() 0
#endif // USE_ROCM

// cuSparse Generic API spsv function was added in CUDA 11.3.0
#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && (CUSPARSE_VERSION >= 11500)
#define AT_USE_CUSPARSE_GENERIC_SPSV() 1
#else
#define AT_USE_CUSPARSE_GENERIC_SPSV() 0
#endif

// cuSparse Generic API spsm function was added in CUDA 11.3.1
#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && (CUSPARSE_VERSION >= 11600)
#define AT_USE_CUSPARSE_GENERIC_SPSM() 1
#else
#define AT_USE_CUSPARSE_GENERIC_SPSM() 0
#endif

// cuSparse Generic API sddmm function was added in CUDA 11.2.1 (cuSparse version 11400)
#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && (CUSPARSE_VERSION >= 11400)
#define AT_USE_CUSPARSE_GENERIC_SDDMM() 1
#else
#define AT_USE_CUSPARSE_GENERIC_SDDMM() 0
#endif

// BSR triangular solve functions were added in hipSPARSE 1.11.2 (ROCm 4.5.0)
#if defined(CUDART_VERSION) ||                            \
      (defined(USE_ROCM) && ROCM_VERSION >= 40500 )
#define AT_USE_HIPSPARSE_TRIANGULAR_SOLVE() 1
#else
#define AT_USE_HIPSPARSE_TRIANGULAR_SOLVE() 0
#endif
