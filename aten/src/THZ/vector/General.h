#ifndef THZ_VECTOR_GENERAL
#define THZ_VECTOR_GENERAL

#include <complex.h>

#if defined(__AVX__) || defined(__AVX2__)
#define COMPLEX128_TO_M256D(REAL, IMAG) \
  _mm256_set_pd((IMAG), (REAL), (IMAG), (REAL))

#define COMPLEX64_TO_M256(REAL, IMAG) \
  _mm256_set_ps((IMAG), (REAL), (IMAG), (REAL), (IMAG), (REAL), (IMAG), (REAL))

#endif

#ifdef __AVX2__

/**
 * 1. exchange
 * 2. make real and imag in correct order
 */
#define MAKE_COMPLEX128(YMM0, YMM1, REAL, IMAG) \
  YMM0 = _mm256_permute2f128_pd(REAL, IMAG, 32); \
  YMM1 = _mm256_permute2f128_pd(REAL, IMAG, 49); \
  YMM0 = _mm256_permute4x64_pd(YMM0, 216); \
  YMM1 = _mm256_permute4x64_pd(YMM1, 216);

// 1. exchange
// each of following contains two complex number
// 2. make real and imag in correct order
// 2.1 exchange inner 64bit first
// 2.2 exchange in 128bit lane

#define MAKE_COMPLEX64(YMM0, YMM1, REAL, IMAG) \
  YMM0 = _mm256_permute2f128_ps(REAL, IMAG, 32); \
  YMM1 = _mm256_permute2f128_ps(REAL, IMAG, 49); \
  YMM0 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(YMM0), 216)); \
  YMM1 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(YMM1), 216)); \
  YMM0 = _mm256_permute_ps(YMM0, 216); \
  YMM1 = _mm256_permute_ps(YMM1, 216);

#define HSUB_MUL_INORDER_PD(TARGET, YMM0, YMM1, YMM2, YMM3) \
  TARGET = _mm256_hsub_pd(_mm256_mul_pd(YMM0, YMM2), \
    _mm256_mul_pd(YMM1, YMM3)); \
  TARGET = _mm256_permute4x64_pd(TARGET, 216);

#define HADD_MUL_INORDER_PD(TARGET, YMM0, YMM1, YMM2, YMM3) \
  TARGET = _mm256_hadd_pd(_mm256_mul_pd(YMM0, YMM2), \
    _mm256_mul_pd(YMM1, YMM3)); \
  TARGET = _mm256_permute4x64_pd(TARGET, 216);

#define HSUB_MUL_INORDER_PS(TARGET, YMM0, YMM1, YMM2, YMM3) \
  TARGET = _mm256_hsub_ps(_mm256_mul_ps(YMM0, YMM2), \
    _mm256_mul_ps(YMM1, YMM3)); \
  TARGET = _mm256_castpd_ps( \
    _mm256_permute4x64_pd( \
      _mm256_castps_pd(TARGET), 216));

#define HADD_MUL_INORDER_PS(TARGET, YMM0, YMM1, YMM2, YMM3) \
  TARGET = _mm256_hadd_ps(_mm256_mul_ps(YMM0, YMM2), \
  _mm256_mul_ps(YMM1, YMM3)); \
  TARGET = _mm256_castpd_ps( \
    _mm256_permute4x64_pd( \
      _mm256_castps_pd(TARGET), 216));

#endif

#endif