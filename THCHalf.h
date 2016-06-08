#ifndef THC_HALF_CONVERSION_INC
#define THC_HALF_CONVERSION_INC

#include "THCGeneral.h"

// We compile with CudaHalfTensor support if we have this:
#if CUDA_VERSION >= 7050 || CUDA_HAS_FP16
#define CUDA_HALF_TENSOR 1
#endif

// Native fp16 ALU instructions are available if we have this:
#if defined(CUDA_HALF_TENSOR) && (__CUDA_ARCH__ >= 530)
#define CUDA_HALF_INSTRUCTIONS 1
#endif

#ifdef CUDA_HALF_TENSOR

#include <cuda_fp16.h>
#include <stdint.h>

THC_EXTERNC void THCFloat2Half(THCState *state, half *out, float *in, long len);
THC_EXTERNC void THCHalf2Float(THCState *state, float *out, half *in, long len);
THC_EXTERNC half THC_float2half(float a);
THC_EXTERNC float THC_half2float(half a);

#endif // CUDA_HALF_TENSOR

#endif
