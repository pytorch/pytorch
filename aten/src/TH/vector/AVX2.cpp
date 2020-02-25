#if defined(__AVX2__)
#ifndef _MSC_VER
#include <x86intrin.h>
#else
#include <intrin.h>
#include <immintrin.h>
#endif
#include <TH/vector/AVX2.h>
#include <ATen/native/cpu/avx_mathfun.h>
#include <ATen/Context.h>

void THDoubleVector_cadd_AVX2(double *z, const double *x, const double *y, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256d YMM15 = _mm256_set_pd(c, c, c, c);
  __m256d YMM0, YMM1, YMM2, YMM3;
  for (i=0; i<=((n)-8); i+=8) {
    YMM0 = _mm256_loadu_pd(y+i);
    YMM1 = _mm256_loadu_pd(y+i+4);
    YMM2 = _mm256_loadu_pd(x+i);
    YMM3 = _mm256_loadu_pd(x+i+4);
    YMM2 = _mm256_fmadd_pd(YMM0, YMM15, YMM2);
    YMM3 = _mm256_fmadd_pd(YMM1, YMM15, YMM3);
    _mm256_storeu_pd(z+i, YMM2);
    _mm256_storeu_pd(z+i+4, YMM3);
  }
  for (; i<(n); i++) {
    z[i] = x[i] + y[i] * c;
  }
}

void THFloatVector_cadd_AVX2(float *z, const float *x, const float *y, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256 YMM15 = _mm256_set_ps(c, c, c, c, c, c, c, c);
  __m256 YMM0, YMM1, YMM2, YMM3;
  for (i=0; i<=((n)-16); i+=16) {
    YMM0 = _mm256_loadu_ps(y+i);
    YMM1 = _mm256_loadu_ps(y+i+8);
    YMM2 = _mm256_loadu_ps(x+i);
    YMM3 = _mm256_loadu_ps(x+i+8);
    YMM2 = _mm256_fmadd_ps(YMM0, YMM15, YMM2);
    YMM3 = _mm256_fmadd_ps(YMM1, YMM15, YMM3);
    _mm256_storeu_ps(z+i, YMM2);
    _mm256_storeu_ps(z+i+8, YMM3);
  }
  for (; i<(n); i++) {
    z[i] = x[i] + y[i] * c;
  }
}

#endif // defined(__AVX2__)
