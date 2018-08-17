#ifndef THC_HALF_CONVERSION_INC
#define THC_HALF_CONVERSION_INC

#include "THCGeneral.h"

#include <cuda_fp16.h>
#include <stdint.h>

#if CUDA_VERSION >= 9000 || defined(__HIP_PLATFORM_HCC__)
#ifndef __cplusplus
typedef __half_raw half;
#endif
#endif

THC_EXTERNC void THCFloat2Half(THCState *state, half *out, float *in, ptrdiff_t len);
THC_EXTERNC void THCHalf2Float(THCState *state, float *out, half *in, ptrdiff_t len);
THC_API half THC_float2half(float a);
THC_API float THC_half2float(half a);

/* Check for native fp16 support on the current device (CC 5.3+) */
THC_API int THC_nativeHalfInstructions(THCState *state);

/* Check for performant native fp16 support on the current device */
THC_API int THC_fastHalfInstructions(THCState *state);

#endif
