#include "THCHalf.h"
#include "THCThrustAllocator.cuh"
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

struct __half2floatOp {
  __device__ float operator()(half v) { return __half2float(v); }
};

struct __float2halfOp {
  __device__ half operator()(float v) { return __float2half(v); }
};

void THCFloat2Half(THCState *state, half *out, float *in, ptrdiff_t len) {
  THCThrustAllocator thrustAlloc(state);
  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    in, in + len, out, __float2halfOp());
}

void THCHalf2Float(THCState *state, float *out, half *in, ptrdiff_t len) {
  THCThrustAllocator thrustAlloc(state);
  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    in, in + len, out, __half2floatOp());
}

// FixMe: could call TH_half2float
// and convert types here, but maybe slower?
float THC_half2float(half h)
{
  unsigned sign = ((h.x >> 15) & 1);
  unsigned exponent = ((h.x >> 10) & 0x1f);
  unsigned mantissa = ((h.x & 0x3ff) << 13);

  if (exponent == 0x1f) {  /* NaN or Inf */
    mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
    exponent = 0xff;
  } else if (!exponent) {  /* Denorm or Zero */
    if (mantissa) {
      unsigned int msb;
      exponent = 0x71;
      do {
        msb = (mantissa & 0x400000);
        mantissa <<= 1;  /* normalize */
        --exponent;
      } while (!msb);
      mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
    }
  } else {
    exponent += 0x70;
  }

  int temp = ((sign << 31) | (exponent << 23) | mantissa);

  float x;
  memcpy(&x,&temp,sizeof(float));
  return x;
}

half THC_float2half(float f)
{
  half ret;

  unsigned x;
  memcpy(&x,&f,sizeof(f));
  unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
  unsigned sign, exponent, mantissa;

  // Get rid of +NaN/-NaN case first.
  if (u > 0x7f800000) {
    ret.x = 0x7fffU;
    return ret;
  }

  sign = ((x >> 16) & 0x8000);

  // Get rid of +Inf/-Inf, +0/-0.
  if (u > 0x477fefff) {
    ret.x = sign | 0x7c00U;
    return ret;
  }
  if (u < 0x33000001) {
    ret.x = (sign | 0x0000);
    return ret;
  }

  exponent = ((u >> 23) & 0xff);
  mantissa = (u & 0x7fffff);

  if (exponent > 0x70) {
    shift = 13;
    exponent -= 0x70;
  } else {
    shift = 0x7e - exponent;
    exponent = 0;
    mantissa |= 0x800000;
  }
  lsb = (1 << shift);
  lsb_s1 = (lsb >> 1);
  lsb_m1 = (lsb - 1);

  // Round to nearest even.
  remainder = (mantissa & lsb_m1);
  mantissa >>= shift;
  if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
    ++mantissa;
    if (!(mantissa & 0x3ff)) {
      ++exponent;
      mantissa = 0;
    }
  }

  ret.x = (sign | (exponent << 10) | mantissa);
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
