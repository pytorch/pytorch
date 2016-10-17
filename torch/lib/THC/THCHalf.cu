#include "THCHalf.h"
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

struct __half2floatOp {
  __device__ float operator()(half v) { return __half2float(v); }
};

struct __float2halfOp {
  __device__ half operator()(float v) { return __float2half(v); }
};

void THCFloat2Half(THCState *state, half *out, float *in, ptrdiff_t len) {
  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    in, in + len, out, __float2halfOp());
}

void THCHalf2Float(THCState *state, float *out, half *in, ptrdiff_t len) {
  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    in, in + len, out, __half2floatOp());
}

float THC_half2float(half a)
{
  unsigned int bits = a.x & 0x7fff;
  unsigned int sign = a.x & 0x8000;
  unsigned int exp = a.x & 0x7c00;

  bits <<= 13;
  sign <<= 16;

  bits += 0x38000000U;

  // flush denormals to 0
  bits = (exp == 0 ? 0 : bits) | sign;

  union {
    float f;
    unsigned int v;
  } conv;
  conv.v = bits;

  return conv.f;
}

/*
  Copyright (c) 2015, Norbert Juffa
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

half THC_float2half(float a)
{
  uint32_t ia;
  uint16_t ir;
  memcpy(&ia, &a, sizeof(float));

  ir = (ia >> 16) & 0x8000;
  if ((ia & 0x7f800000) == 0x7f800000) {
    if ((ia & 0x7fffffff) == 0x7f800000) {
      ir |= 0x7c00; /* infinity */
    } else {
      ir = 0x7fff; /* canonical NaN */
    }
  } else if ((ia & 0x7f800000) >= 0x33000000) {
    int shift = (int)((ia >> 23) & 0xff) - 127;
    if (shift > 15) {
      ir |= 0x7c00; /* infinity */
    } else {
      ia = (ia & 0x007fffff) | 0x00800000; /* extract mantissa */
      if (shift < -14) { /* denormal */
        ir |= ia >> (-1 - shift);
        ia = ia << (32 - (-1 - shift));
      } else { /* normal */
        ir |= ia >> (24 - 11);
        ia = ia << (32 - (24 - 11));
        ir = ir + ((14 + shift) << 10);
      }
      /* IEEE-754 round to nearest of even */
      if ((ia > 0x80000000) || ((ia == 0x80000000) && (ir & 1))) {
        ir++;
      }
    }
  }

  half ret;
  memcpy(&ret, &ir, sizeof(half));
  return ret;
}

THC_EXTERNC int THC_nativeHalfInstructions(THCState *state) {
  cudaDeviceProp* prop =
    THCState_getCurrentDeviceProperties(state);

  // CC 5.3+
  return (prop->major > 5 ||
          (prop->major == 5 && prop->minor == 3));
}

THC_EXTERNC int THC_fastHalfInstructions(THCState *state) {
  cudaDeviceProp* prop =
    THCState_getCurrentDeviceProperties(state);

  // Check for CC 6.0 only (corresponds to P100)
  return (prop->major == 6 && prop->minor == 0);
}
