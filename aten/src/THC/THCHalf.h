#ifndef THC_HALF_CONVERSION_INC
#define THC_HALF_CONVERSION_INC

#include "THCGeneral.h"

/* We compile with CudaHalfTensor support if we have this: */
#if CUDA_VERSION >= 7050 || CUDA_HAS_FP16 || defined(__HIP_PLATFORM_HCC__)
#define CUDA_HALF_TENSOR 1
#endif

#ifdef CUDA_HALF_TENSOR

#include <cuda_fp16.h>
#include <stdint.h>

#if CUDA_VERSION >= 9000 || defined(__HIP_PLATFORM_HCC__)
#ifndef __cplusplus
typedef __half_raw half;
#endif
#endif

THC_API half THC_float2half(float a);
THC_API float THC_half2float(half a);

#endif /* CUDA_HALF_TENSOR */

#endif
