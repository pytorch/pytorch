#ifndef THC_HALF_CONVERSION_INC
#define THC_HALF_CONVERSION_INC

#include "THCGeneral.h"

#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __cplusplus
#include <ATen/core/Half.h>
#endif

#ifdef __cplusplus
#define THCHalf at::Half
#else
typedef struct at_Half at_Half;
#define THCHalf at_Half
#endif

#if CUDA_VERSION >= 9000 || defined(__HIP_PLATFORM_HCC__)
#ifndef __cplusplus
typedef __half_raw half;
#endif
#endif

THC_API half THC_float2half(float a);
THC_API float THC_half2float(half a);

#endif
