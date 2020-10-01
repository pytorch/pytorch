#if defined(__AVX__)
#ifndef _MSC_VER
#include <x86intrin.h>
#else
#include <intrin.h>
#endif

#include <TH/vector/AVX.h>
#include <TH/THGeneral.h>

void THDoubleVector_fill_AVX(double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m256d YMM0 = _mm256_set_pd(c, c, c, c);
  for (i=0; i<=((n)-16); i+=16) {
    _mm256_storeu_pd((x)+i  , YMM0);
    _mm256_storeu_pd((x)+i+4, YMM0);
    _mm256_storeu_pd((x)+i+8, YMM0);
    _mm256_storeu_pd((x)+i+12, YMM0);
  }
  off = (n) - ((n)%16);
  for (i=0; i<((n)%16); i++) {
    x[off+i] = c;
  }
}

void THDoubleVector_muls_AVX(double *y, const double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256d YMM15 = _mm256_set_pd(c, c, c, c);
  __m256d YMM0, YMM1;
  for (i=0; i<=((n)-8); i+=8) {
    YMM0 = _mm256_loadu_pd(x+i);
    YMM1 = _mm256_loadu_pd(x+i+4);
    YMM0 = _mm256_mul_pd(YMM0, YMM15);
    YMM1 = _mm256_mul_pd(YMM1, YMM15);
    _mm256_storeu_pd(y+i, YMM0);
    _mm256_storeu_pd(y+i+4, YMM1);
  }
  for (; i<n; i++) {
    y[i] = x[i] * c;
  }
}

void THFloatVector_fill_AVX(float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m256 YMM0 = _mm256_set_ps(c, c, c, c, c, c, c, c);
  for (i=0; i<=((n)-32); i+=32) {
    _mm256_storeu_ps((x)+i  , YMM0);
    _mm256_storeu_ps((x)+i+8, YMM0);
    _mm256_storeu_ps((x)+i+16, YMM0);
    _mm256_storeu_ps((x)+i+24, YMM0);
  }
  off = (n) - ((n)%32);
  for (i=0; i<((n)%32); i++) {
    x[off+i] = c;
  }
}

void THFloatVector_muls_AVX(float *y, const float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256 YMM15 = _mm256_set_ps(c, c, c, c, c, c, c, c);
  __m256 YMM0, YMM1;
  for (i=0; i<=((n)-16); i+=16) {
    YMM0 = _mm256_loadu_ps(x+i);
    YMM1 = _mm256_loadu_ps(x+i+8);
    YMM0 = _mm256_mul_ps(YMM0, YMM15);
    YMM1 = _mm256_mul_ps(YMM1, YMM15);
    _mm256_storeu_ps(y+i, YMM0);
    _mm256_storeu_ps(y+i+8, YMM1);
  }
  for (; i<n; i++) {
    y[i] = x[i] * c;
  }
}

#endif // defined(__AVX__)
