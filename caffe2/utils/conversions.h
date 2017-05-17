#pragma once

#include <caffe2/core/types.h>

#ifdef __CUDA_ARCH__
#include <cuda_fp16.h>
#endif

#ifdef __CUDA_ARCH__
#define CONVERSIONS_DECL __host__ __device__ inline
#else
#define CONVERSIONS_DECL inline
#endif

namespace caffe2 {

namespace convert {

namespace {
inline float16 cpu_float2half_rn(float f) {
  float16 ret;

  static_assert(
      sizeof(unsigned int) == sizeof(float),
      "Programming error sizeof(unsigned int) != sizeof(float)");

  unsigned* xp = reinterpret_cast<unsigned int*>(&f);
  unsigned x = *xp;
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

inline float cpu_half2float(float16 h) {
  unsigned sign = ((h.x >> 15) & 1);
  unsigned exponent = ((h.x >> 10) & 0x1f);
  unsigned mantissa = ((h.x & 0x3ff) << 13);

  if (exponent == 0x1f) { /* NaN or Inf */
    mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
    exponent = 0xff;
  } else if (!exponent) { /* Denorm or Zero */
    if (mantissa) {
      unsigned int msb;
      exponent = 0x71;
      do {
        msb = (mantissa & 0x400000);
        mantissa <<= 1; /* normalize */
        --exponent;
      } while (!msb);
      mantissa &= 0x7fffff; /* 1.mantissa is implicit */
    }
  } else {
    exponent += 0x70;
  }

  unsigned i = ((sign << 31) | (exponent << 23) | mantissa);
  float ret;
  memcpy(&ret, &i, sizeof(i));
  return ret;
}

}; // anonymous
// general version: defer to static_cast
template <typename IN, typename OUT>
CONVERSIONS_DECL OUT To(const IN in) {
  return static_cast<OUT>(in);
}

#if __CUDA_ARCH__
__device__ __inline__ __half inf_clip(__half h) {
  int isi = __hisinf(h);
  if (isi > 0) {
    // Exponent all ones except LSB (0x1e), mantissa is all ones (0x3ff)
    h.x = 0x7bffU;
  } else if (isi < 0) {
    // As above, negated
    h.x = 0x7bffU ^ 0x8000;
  }
  return h;
}
#endif

// explicit for fp16
template <>
CONVERSIONS_DECL float16 To(const float in) {
#if __CUDA_ARCH__
  // hacky interface between C2 fp16 and CUDA
  float16 ret;
#if 0
  // alternative truncation scheme
  __half r;
  r.x = __float2half_rn(in);
  ret.x = inf_clip(r).x;
#else
  ret.x = __float2half(in).x;
#endif
  return ret;
#else
  return cpu_float2half_rn(in);
#endif
}

template <>
CONVERSIONS_DECL float To(const float16 in) {
#if __CUDA_ARCH__
  __half tmp;
  tmp.x = in.x;
  return __half2float(tmp);
#else
  return cpu_half2float(in);
#endif
};

template <>
CONVERSIONS_DECL float To(const float in) {
  return in;
}

template <typename OUT, typename IN>
CONVERSIONS_DECL OUT Get(IN x) {
  return static_cast<OUT>(x);
}

template <>
CONVERSIONS_DECL float Get(float16 x) {
  return To<float16, float>(x);
}

template <>
CONVERSIONS_DECL float16 Get(float x) {
  return To<float, float16>(x);
}

}; // namespace convert

}; // namespace caffe2

#undef CONVERSIONS_DECL
