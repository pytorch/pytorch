#if defined(__AVX2__)
#ifndef _MSC_VER
#include <x86intrin.h>
#else
#include <intrin.h>
#include <immintrin.h>
#endif
#include "AVX2.h"
#include "avx_mathfun.h"
#include "../THRandom.h"

#include <assert.h>

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

static void normal_fill_16_AVX2(float *normal,
                               const float *uniform,
                               const __m256* two_pi,
                               const __m256* one,
                               const __m256* minus_two) {
  const __m256 u1 = _mm256_sub_ps(*one, _mm256_load_ps(uniform));
  const __m256 u2 = _mm256_load_ps(uniform + 8);

  __m256 radius = _mm256_sqrt_ps(_mm256_mul_ps(*minus_two, log256_ps(u1)));
  __m256 theta = _mm256_mul_ps(*two_pi, u2);

  __m256 sintheta, costheta;
  sincos256_ps(theta, &sintheta, &costheta);

  _mm256_storeu_ps(normal, _mm256_mul_ps(radius, costheta));
  _mm256_storeu_ps(normal + 8, _mm256_mul_ps(radius, sintheta));
}

void THFloatTensor_normal_fill_AVX2(float *data,
                                    const int size,
                                    THGenerator *generator,
                                    const float mean,
                                    const float stdv)
{
  assert(size >= 16 && "Tensor size must be >= 16 for AVX2 normal fill");
  const __m256 two_pi = _mm256_set1_ps(2.0f * 3.14159265358979323846f);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 minus_two = _mm256_set1_ps(-2.0f);

  // First fill the data with the uniform numbers. Box-Mueller is a 2 -> 2
  // mapping of 2 uniform numbers to 2 normal numbers (per iteration), so we
  // we need exactly as much space for uniform and normal numbers and can just
  // use the single buffer for both.
  float tail[16];
  for (int i = 0; i < size; ++i) {
    data[i] = THRandom_uniformFloat(generator, 0, 1);
    // Store the last 16 uniform values we produce for later edge cases.
    if (size % 16 != 0 && i >= (size - 16)) {
      tail[i - (size - 16)] = data[i];
    }
  }

  for (int i = 0; i < size - 16; i += 16) {
    normal_fill_16_AVX2(data + i, data + i, &two_pi, &one, &minus_two);
  }

  if (size % 16 != 0) {
    // We essentially rewind so that we have 16 values, including the tail which
    // we didn't handle above, and then compute them all in one step.
    normal_fill_16_AVX2(data + (size - 16), tail, &two_pi, &one, &minus_two);
  }
}

#endif // defined(__AVX2__)
