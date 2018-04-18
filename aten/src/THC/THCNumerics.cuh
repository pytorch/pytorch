#ifndef THC_NUMERICS_INC
#define THC_NUMERICS_INC

#include <cuda.h>
#include <limits.h>
#include <assert.h>
#include "THCHalf.h"

/// Class for numeric limits of the particular data type, which
/// includes support for `half`.
/// Unfortunately since `half` does not have a constructor, these have
/// to be expressed as functions (either that or non-const statics).
template <typename T>
struct THCNumerics {
};

template <typename scalar_t>
static inline __host__ __device__ scalar_t powi(scalar_t a, scalar_t b) {
  assert(THCNumerics<scalar_t>::ge(b, 0));
  scalar_t result = 1;
  while (b) {
    if (b & 1) {
       result *= a;
    }
    b /= 2;
    a *= a;
  }
  return result;
}

template <>
struct THCNumerics<uint8_t> {
  static inline __host__ __device__ uint8_t min() { return 0; }
  static inline __host__ __device__ uint8_t max() { return UCHAR_MAX; }

  static inline __host__ __device__ bool lt(uint8_t a, uint8_t b) { return a < b; }
  static inline __host__ __device__ bool le(uint8_t a, uint8_t b) { return a <= b; }
  static inline __host__ __device__ bool gt(uint8_t a, uint8_t b) { return a > b; }
  static inline __host__ __device__ bool ge(uint8_t a, uint8_t b) { return a >= b; }
  static inline __host__ __device__ bool eq(uint8_t a, uint8_t b) { return a == b; }
  static inline __host__ __device__ bool ne(uint8_t a, uint8_t b) { return a != b; }

  static inline __host__ __device__  uint8_t neg(int8_t a) { return -a; }
  static inline __host__ __device__  uint8_t add(uint8_t a, uint8_t b) { return a + b; }
  static inline __host__ __device__  uint8_t mul(uint8_t a, uint8_t b) { return a * b; }
  static inline __host__ __device__  uint8_t sub(uint8_t a, uint8_t b) { return a - b; }
  static inline __host__ __device__  uint8_t div(uint8_t a, uint8_t b) { return a / b; }
  static inline __host__ __device__  uint8_t abs(uint8_t a) { return a; }
  static inline __host__ __device__  uint8_t pow(uint8_t a, uint8_t b) { return powi<uint8_t>(a, b); }
};

template <>
struct THCNumerics<int8_t> {
  static inline __host__ __device__ int8_t min() { return SCHAR_MIN; }
  static inline __host__ __device__ int8_t max() { return SCHAR_MAX; }

  static inline __host__ __device__ bool lt(int8_t a, int8_t b) { return a < b; }
  static inline __host__ __device__ bool le(int8_t a, int8_t b) { return a <= b; }
  static inline __host__ __device__ bool gt(int8_t a, int8_t b) { return a > b; }
  static inline __host__ __device__ bool ge(int8_t a, int8_t b) { return a >= b; }
  static inline __host__ __device__ bool eq(int8_t a, int8_t b) { return a == b; }
  static inline __host__ __device__ bool ne(int8_t a, int8_t b) { return a != b; }

  static inline __host__ __device__  int8_t neg(int8_t a) { return -a; }
  static inline __host__ __device__  int8_t add(int8_t a, int8_t b) { return a + b; }
  static inline __host__ __device__  int8_t mul(int8_t a, int8_t b) { return a * b; }
  static inline __host__ __device__  int8_t sub(int8_t a, int8_t b) { return a - b; }
  static inline __host__ __device__  int8_t div(int8_t a, int8_t b) { return a / b; }
  static inline __host__ __device__  int8_t abs(int8_t a) { return ::abs((int)a); }
  static inline __host__ __device__  int8_t pow(int8_t a, int8_t b) { return powi<int8_t>(a, b); }
};

template <>
struct THCNumerics<int16_t> {
  static inline __host__ __device__ int16_t min() { return SHRT_MIN; }
  static inline __host__ __device__ int16_t max() { return SHRT_MAX; }

  static inline __host__ __device__ bool lt(int16_t a, int16_t b) { return a < b; }
  static inline __host__ __device__ bool le(int16_t a, int16_t b) { return a <= b; }
  static inline __host__ __device__ bool gt(int16_t a, int16_t b) { return a > b; }
  static inline __host__ __device__ bool ge(int16_t a, int16_t b) { return a >= b; }
  static inline __host__ __device__ bool eq(int16_t a, int16_t b) { return a == b; }
  static inline __host__ __device__ bool ne(int16_t a, int16_t b) { return a != b; }

  static inline __host__ __device__  int16_t neg(int16_t a) { return -a; }
  static inline __host__ __device__  int16_t add(int16_t a, int16_t b) { return a + b; }
  static inline __host__ __device__  int16_t mul(int16_t a, int16_t b) { return a * b; }
  static inline __host__ __device__  int16_t sub(int16_t a, int16_t b) { return a - b; }
  static inline __host__ __device__  int16_t div(int16_t a, int16_t b) { return a / b; }
  static inline __host__ __device__  int16_t abs(int16_t a) { return ::abs((int)a); }
  static inline __host__ __device__  int16_t pow(int16_t a, int16_t b) { return powi<int16_t>(a, b); }
};

template <>
struct THCNumerics<int32_t> {
  static inline __host__ __device__ int32_t min() { return INT_MIN; }
  static inline __host__ __device__ int32_t max() { return INT_MAX; }

  static inline __host__ __device__ bool lt(int32_t a, int32_t b) { return a < b; }
  static inline __host__ __device__ bool le(int32_t a, int32_t b) { return a <= b; }
  static inline __host__ __device__ bool gt(int32_t a, int32_t b) { return a > b; }
  static inline __host__ __device__ bool ge(int32_t a, int32_t b) { return a >= b; }
  static inline __host__ __device__ bool eq(int32_t a, int32_t b) { return a == b; }
  static inline __host__ __device__ bool ne(int32_t a, int32_t b) { return a != b; }

  static inline __host__ __device__  int32_t neg(int32_t a) { return -a; }
  static inline __host__ __device__  int32_t add(int32_t a, int32_t b) { return a + b; }
  static inline __host__ __device__  int32_t mul(int32_t a, int32_t b) { return a * b; }
  static inline __host__ __device__  int32_t sub(int32_t a, int32_t b) { return a - b; }
  static inline __host__ __device__  int32_t div(int32_t a, int32_t b) { return a / b; }
  static inline __host__ __device__  int32_t abs(int32_t a) { return ::abs(a); }
  static inline __host__ __device__  int32_t pow(int32_t a, int32_t b) { return powi<int32_t>(a, b); }
};

template <>
struct THCNumerics<int64_t> {
#ifdef _MSC_VER
  static inline __host__ __device__ int64_t min() { return _I64_MIN; }
  static inline __host__ __device__ int64_t max() { return _I64_MAX; }
#else
  static inline __host__ __device__ int64_t min() { return LONG_MIN; }
  static inline __host__ __device__ int64_t max() { return LONG_MAX; }
#endif

  static inline __host__ __device__ bool lt(int64_t a, int64_t b) { return a < b; }
  static inline __host__ __device__ bool le(int64_t a, int64_t b) { return a <= b; }
  static inline __host__ __device__ bool gt(int64_t a, int64_t b) { return a > b; }
  static inline __host__ __device__ bool ge(int64_t a, int64_t b) { return a >= b; }
  static inline __host__ __device__ bool eq(int64_t a, int64_t b) { return a == b; }
  static inline __host__ __device__ bool ne(int64_t a, int64_t b) { return a != b; }


  static inline __host__ __device__  int64_t neg(int64_t a) { return -a; }
  static inline __host__ __device__  int64_t add(int64_t a, int64_t b) { return a + b; }
  static inline __host__ __device__  int64_t mul(int64_t a, int64_t b) { return a * b; }
  static inline __host__ __device__  int64_t sub(int64_t a, int64_t b) { return a - b; }
  static inline __host__ __device__  int64_t div(int64_t a, int64_t b) { return a / b; };
  static inline __host__ __device__  int64_t abs(int64_t a) { return labs(a); }
  static inline __host__ __device__  int64_t pow(int64_t a, int64_t b) { return powi<int64_t>(a, b); }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct THCNumerics<half> {
#if CUDA_VERSION < 9000
  static inline __host__ __device__ half min() { half h; h.x = 0xfbff; return h; }
  static inline __host__ __device__ half max() { half h; h.x = 0x7bff; return h; }
#else
  static inline __host__ __device__ half min() { __half_raw h; h.x = 0xfbff; return h; }
  static inline __host__ __device__ half max() { __half_raw h; h.x = 0x7bff; return h; }
#endif

  static inline __host__ __device__ bool lt(half a, half b) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hlt(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return fa < fb;
#endif
#else // __CUDA_ARCH__
    return THC_half2float(a) < THC_half2float(b);
#endif
  }

  static inline __host__ __device__ bool le(half a, half b) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hle(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return fa <= fb;
#endif
#else // __CUDA_ARCH__
    return THC_half2float(a) <= THC_half2float(b);
#endif
  }

  static inline __host__ __device__ bool gt(half a, half b) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hgt(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return fa > fb;
#endif
#else // __CUDA_ARCH__
    return THC_half2float(a) > THC_half2float(b);
#endif
  }

  static inline __host__ __device__ bool ge(half a, half b) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hge(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return fa >= fb;
#endif
#else // __CUDA_ARCH__
    return THC_half2float(a) >= THC_half2float(b);
#endif
  }

  static inline __host__ __device__ bool eq(half a, half b) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __heq(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return fa == fb;
#endif
#else // __CUDA_ARCH__
    return THC_half2float(a) == THC_half2float(b);
#endif
  }

  static inline __host__ __device__ bool ne(half a, half b) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hne(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return fa != fb;
#endif
#else // __CUDA_ARCH__
    return THC_half2float(a) != THC_half2float(b);
#endif
  }

  static inline __host__ __device__ half exp(half a) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return hexp(a);
#else
    float fa = __half2float(a);
    return __float2half(expf(fa));
#endif
#else // __CUDA_ARCH__
    return THC_float2half(expf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half exp10(half a) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return hexp10(a);
#else
    float fa = __half2float(a);
    return __float2half(exp10f(fa));
#endif
#else // __CUDA_ARCH__
    return THC_float2half(exp10f(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half log(half a) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return hlog(a);
#else
    float fa = __half2float(a);
    return __float2half(logf(fa));
#endif
#else // __CUDA_ARCH__
    return THC_float2half(logf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half log10(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(log10f(fa));
#else // __CUDA_ARCH__
    return THC_float2half(log10f(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half log1p(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(log1pf(fa));
#else // __CUDA_ARCH__
    return THC_float2half(log1pf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half log2(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(log2f(fa));
#else // __CUDA_ARCH__
    return THC_float2half(log2f(THC_half2float(a)));
#endif
  }

static inline __host__ __device__ half lgamma(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(lgammaf(fa));
#else // __CUDA_ARCH__
    return THC_float2half(lgammaf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half expm1(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(expm1f(fa));
#else // __CUDA_ARCH__
    return THC_float2half(expm1f(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half cos(half a) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return hcos(a);
#else
    float fa = __half2float(a);
    return __float2half(cosf(fa));
#endif
#else // __CUDA_ARCH__
    return THC_float2half(cosf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half sin(half a) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return hsin(a);
#else
    float fa = __half2float(a);
    return __float2half(sinf(fa));
#endif
#else // __CUDA_ARCH__
    return THC_float2half(sinf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half sqrt(half a) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return hsqrt(a);
#else
    float fa = __half2float(a);
    return __float2half(sqrtf(fa));
#endif
#else // __CUDA_ARCH__
    return THC_float2half(sqrtf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half rsqrt(half a) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return hrsqrt(a);
#else
    float fa = __half2float(a);
    return __float2half(rsqrtf(fa));
#endif
#else // __CUDA_ARCH__
    return THC_float2half(rsqrtf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half ceil(half a) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return hceil(a);
#else
    float fa = __half2float(a);
    return __float2half(ceilf(fa));
#endif
#else // __CUDA_ARCH__
    return THC_float2half(ceilf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half floor(half a) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return hfloor(a);
#else
    float fa = __half2float(a);
    return __float2half(floorf(fa));
#endif
#else // __CUDA_ARCH__
    return THC_float2half(floorf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half trunc(half a) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return htrunc(a);
#else
    float fa = __half2float(a);
    return __float2half(truncf(fa));
#endif
#else // __CUDA_ARCH__
    return THC_float2half(truncf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half neg(half a) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hneg(a);
#else
    float fa = __half2float(a);
    return __float2half(-fa);
#endif
#else // __CUDA_ARCH__
    return THC_float2half(-(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half acos(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(acosf(fa));
#else // __CUDA_ARCH__
    return THC_float2half(acosf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half cosh(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(coshf(fa));
#else // __CUDA_ARCH__
    return THC_float2half(coshf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half asin(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(asinf(fa));
#else // __CUDA_ARCH__
    return THC_float2half(asinf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half sinh(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(sinhf(fa));
#else // __CUDA_ARCH__
    return THC_float2half(sinhf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half tan(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(tanf(fa));
#else // __CUDA_ARCH__
    return THC_float2half(tanf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half atan(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(atanf(fa));
#else // __CUDA_ARCH__
    return THC_float2half(atanf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half tanh(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(tanhf(fa));
#else // __CUDA_ARCH__
    return THC_float2half(tanhf(THC_half2float(a)));
#endif
  }


   static inline __host__ __device__ half erf(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(erff(fa));
#else // __CUDA_ARCH__
    return THC_float2half(erff(THC_half2float(a)));
#endif
  }


   static inline __host__ __device__ half erfinv(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(erfinvf(fa));
#else // __CUDA_ARCH__
    return THC_float2half(erfinvf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half abs(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(fabs(fa));
#else // __CUDA_ARCH__
    return THC_float2half(fabs(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half round(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(roundf(fa));
#else // __CUDA_ARCH__
    return THC_float2half(roundf(THC_half2float(a)));
#endif
  }

  static inline __host__ __device__ half frac(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(fa - truncf(fa));
#else // __CUDA_ARCH__
    float fa = THC_half2float(a);
    return THC_float2half(fa - floorf(fa));
#endif
  }

  static inline __host__ __device__ half cinv(half a) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    return __float2half(1.0f / fa);
#else // __CUDA_ARCH__
    return THC_float2half(1.0f / THC_half2float(a));
#endif
  }

  static inline __host__ __device__ half add(half a, half b) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hadd(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return __float2half( fa + fb );
#endif
#else // __CUDA_ARCH__
    return THC_float2half(THC_half2float(a) + THC_half2float(b));
#endif
  }

  static inline __host__ __device__ half div(half a, half b) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    float fb = __half2float(b);
    return __float2half( fa / fb );
#else // __CUDA_ARCH__
    return THC_float2half(THC_half2float(a) / THC_half2float(b));
#endif
  }

  static inline __host__ __device__ half mul(half a, half b) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hmul(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return __float2half( fa * fb );
#endif
#else // __CUDA_ARCH__
    return THC_float2half(THC_half2float(a) * THC_half2float(b));
#endif
  }

  static inline __host__ __device__ half sub(half a, half b) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hsub(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return __float2half( fa - fb );
#endif
#else // __CUDA_ARCH__
    return THC_float2half(THC_half2float(a) - THC_half2float(b));
#endif
  }

  static inline __host__ __device__ half pow(half a, half b) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    float fb = __half2float(b);
    return __float2half(powf(fa, fb));
#else // __CUDA_ARCH__
    return THC_float2half(powf(THC_half2float(a), THC_half2float(b)));
#endif
  }

  static inline __host__ __device__ half atan2(half a, half b) {
#ifdef __CUDA_ARCH__
    float fa = __half2float(a);
    float fb = __half2float(b);
    return __float2half(atan2f(fa, fb));
#else // __CUDA_ARCH__
    return THC_float2half(atan2f(THC_half2float(a), THC_half2float(b)));
#endif
  }

};
#endif

template <>
struct THCNumerics<float> {
  static inline __host__ __device__ float min() { return -FLT_MAX; }
  static inline __host__ __device__ float max() { return FLT_MAX; }

  static inline __host__ __device__ bool lt(float a, float b) { return a < b; }
  static inline __host__ __device__ bool le(float a, float b) { return a <= b; }
  static inline __host__ __device__ bool gt(float a, float b) { return a > b; }
  static inline __host__ __device__ bool ge(float a, float b) { return a >= b; }
  static inline __host__ __device__ bool eq(float a, float b) { return a == b; }
  static inline __host__ __device__ bool ne(float a, float b) { return a != b; }

  static inline __host__ __device__  float lgamma(float a) { return lgammaf(a);}
  static inline __host__ __device__  float erfinv(float a) { return erfinvf(a);}
  static inline __host__ __device__  float exp  (float a) { return   expf(a); }
  static inline __host__ __device__  float exp10(float a) { return exp10f(a); }
  static inline __host__ __device__  float log  (float a) { return   logf(a); }
  static inline __host__ __device__  float log10(float a) { return log10f(a); }
  static inline __host__ __device__  float log1p(float a) { return log1pf(a); }
  static inline __host__ __device__  float log2 (float a) { return  log2f(a); }
  static inline __host__ __device__  float expm1(float a) { return expm1f(a); }
  static inline __host__ __device__  float cos  (float a) { return   cosf(a); }
  static inline __host__ __device__  float sin  (float a) { return   sinf(a); }
  static inline __host__ __device__  float sqrt (float a) { return  sqrtf(a); }
  static inline __host__ __device__  float rsqrt(float a) { return rsqrtf(a); }
  static inline __host__ __device__  float ceil (float a) { return  ceilf(a); }
  static inline __host__ __device__  float floor(float a) { return floorf(a); }
  static inline __host__ __device__  float trunc(float a) { return truncf(a); }
  static inline __host__ __device__  float neg  (float a) { return        -a; }
  static inline __host__ __device__  float acos (float a) { return  acosf(a); }
  static inline __host__ __device__  float cosh (float a) { return  coshf(a); }
  static inline __host__ __device__  float acosh(float a) { return acoshf(a); }
  static inline __host__ __device__  float asin (float a) { return  asinf(a); }
  static inline __host__ __device__  float sinh (float a) { return  sinhf(a); }
  static inline __host__ __device__  float asinh(float a) { return asinhf(a); }
  static inline __host__ __device__  float tan  (float a) { return   tanf(a); }
  static inline __host__ __device__  float atan (float a) { return  atanf(a); }
  static inline __host__ __device__  float tanh (float a) { return  tanhf(a); }
  static inline __host__ __device__  float erf  (float a) { return   erff(a); }
  static inline __host__ __device__  float abs  (float a) { return   fabs(a); }
  static inline __host__ __device__  float round(float a) { return roundf(a); }
  static inline __host__ __device__  float frac (float a) { return a - truncf(a); }
  static inline __host__ __device__  float cinv (float a) { return 1.0f / a; }
  static inline __host__ __device__  float add  (float a, float b) { return a + b; }
  static inline __host__ __device__  float div  (float a, float b) { return a / b; }
  static inline __host__ __device__  float mul  (float a, float b) { return a * b; }
  static inline __host__ __device__  float sub  (float a, float b) { return a - b; }
  static inline __host__ __device__  float pow  (float a, float b) { return powf(a, b); }
  static inline __host__ __device__  float atan2(float a, float b) { return atan2f(a, b); }
};

template <>
struct THCNumerics<double> {
  static inline __host__ __device__ double min() { return -DBL_MAX; }
  static inline __host__ __device__ double max() { return DBL_MAX; }

  static inline __host__ __device__ bool lt(double a, double b) { return a < b; }
  static inline __host__ __device__ bool le(double a, double b) { return a <= b; }
  static inline __host__ __device__ bool gt(double a, double b) { return a > b; }
  static inline __host__ __device__ bool ge(double a, double b) { return a >= b; }
  static inline __host__ __device__ bool eq(double a, double b) { return a == b; }
  static inline __host__ __device__ bool ne(double a, double b) { return a != b; }

  static inline __host__ __device__  double lgamma(double a) { return ::lgamma(a);}
  static inline __host__ __device__  double erfinv(double a) { return ::erfinv(a);}
  static inline __host__ __device__  double exp  (double a) { return   ::exp(a); }
  static inline __host__ __device__  double exp10(double a) { return ::exp10(a); }
  static inline __host__ __device__  double log  (double a) { return   ::log(a); }
  static inline __host__ __device__  double log10(double a) { return ::log10(a); }
  static inline __host__ __device__  double log1p(double a) { return ::log1p(a); }
  static inline __host__ __device__  double log2 (double a) { return  ::log2(a); }
  static inline __host__ __device__  double expm1(double a) { return ::expm1(a); }
  static inline __host__ __device__  double cos  (double a) { return   ::cos(a); }
  static inline __host__ __device__  double sin  (double a) { return   ::sin(a); }
  static inline __host__ __device__  double sqrt (double a) { return  ::sqrt(a); }
  static inline __host__ __device__  double rsqrt(double a) { return ::rsqrt(a); }
  static inline __host__ __device__  double ceil (double a) { return  ::ceil(a); }
  static inline __host__ __device__  double floor(double a) { return ::floor(a); }
  static inline __host__ __device__  double trunc(double a) { return ::trunc(a); }
  static inline __host__ __device__  double neg  (double a) { return       -a; }
  static inline __host__ __device__  double acos (double a) { return  ::acos(a); }
  static inline __host__ __device__  double cosh (double a) { return  ::cosh(a); }
  static inline __host__ __device__  double acosh(double a) { return ::acosh(a); }
  static inline __host__ __device__  double asin (double a) { return  ::asin(a); }
  static inline __host__ __device__  double sinh (double a) { return  ::sinh(a); }
  static inline __host__ __device__  double asinh(double a) { return ::asinh(a); }
  static inline __host__ __device__  double tan  (double a) { return   ::tan(a); }
  static inline __host__ __device__  double atan (double a) { return  ::atan(a); }
  static inline __host__ __device__  double tanh (double a) { return  ::tanh(a); }
  static inline __host__ __device__  double erf  (double a) { return   ::erf(a); }
  static inline __host__ __device__  double abs  (double a) { return   ::abs(a); }
  static inline __host__ __device__  double round(double a) { return ::round(a); }
  static inline __host__ __device__  double frac (double a) { return a - ::trunc(a); }
  static inline __host__ __device__  double cinv (double a) { return 1.0 / a; }
  static inline __host__ __device__  double add  (double a, double b) { return a + b; }
  static inline __host__ __device__  double div  (double a, double b) { return a / b; }
  static inline __host__ __device__  double mul  (double a, double b) { return a * b; }
  static inline __host__ __device__  double sub  (double a, double b) { return a - b; }
  static inline __host__ __device__  double pow  (double a, double b) { return ::pow(a, b); }
  static inline __host__ __device__  double atan2(double a, double b) { return ::atan2(a, b); }
};

/// `half` has some type conversion issues associated with it, since it
/// is a struct without a constructor/implicit conversion constructor.
/// We use this to convert scalar values to the given type that the
/// tensor expects.
template <typename In, typename Out>
struct ScalarConvert {
  static __host__ __device__ Out to(const In v) { return (Out) v; }
};

#ifdef CUDA_HALF_TENSOR
template <typename Out>
struct ScalarConvert<half, Out> {
  static __host__ __device__ Out to(const half v) {
#ifdef __CUDA_ARCH__
    return (Out) __half2float(v);
#else
    return (Out) THC_half2float(v);
#endif
  }
};

template <typename In>
struct ScalarConvert<In, half> {
  static __host__ __device__ half to(const In v) {
#ifdef __CUDA_ARCH__
    return __float2half((float) v);
#else
    return THC_float2half((float) v);
#endif
  }
};

template <>
struct ScalarConvert<half, half> {
  static __host__ __device__ half to(const half v) {
    return v;
  }
};

template <typename T, typename U>
__host__ __device__ T scalar_cast(U u) {
  return ScalarConvert<U, T>::to(u);
}

#endif

#endif // THC_NUMERICS_INC
