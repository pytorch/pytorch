#if defined(__AVX2__)
#ifndef _MSC_VER
#include <x86intrin.h>
#else
#include <intrin.h>
#include <immintrin.h>
#endif
#include "AVX2.h"
#include <ATen/native/cpu/avx_mathfun.h>
#include "../THRandom.h"

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

static void normal_fill_16_AVX2(float *data,
                                const __m256* two_pi,
                                const __m256* one,
                                const __m256* minus_two,
                                const __m256* mean,
                                const __m256* stddev) {
  const __m256 u1 = _mm256_sub_ps(*one, _mm256_loadu_ps(data));
  const __m256 u2 = _mm256_loadu_ps(data + 8);

  // sincos256_ps and log256_ps are from avx_mathfun.h
  const __m256 radius = _mm256_sqrt_ps(_mm256_mul_ps(*minus_two, log256_ps(u1)));
  const __m256 theta = _mm256_mul_ps(*two_pi, u2);

  __m256 sintheta, costheta;
  sincos256_ps(theta, &sintheta, &costheta);

  const __m256 n1 = _mm256_mul_ps(radius, costheta);
  const __m256 n2 = _mm256_mul_ps(radius, sintheta);

  _mm256_storeu_ps(data, _mm256_fmadd_ps(n1, *stddev, *mean));
  _mm256_storeu_ps(data + 8, _mm256_fmadd_ps(n2, *stddev, *mean));
}

void THFloatVector_normal_fill_AVX2(float *data,
                                    const int64_t size,
                                    THGenerator *generator,
                                    const float mean,
                                    const float stddev)
{
  THAssert(size >= 16 && "Size must be >= 16 for AVX2 normal fill");
  const __m256 two_pi = _mm256_set1_ps(2.0f * M_PI);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 minus_two = _mm256_set1_ps(-2.0f);
  const __m256 mean_v = _mm256_set1_ps(mean);
  const __m256 stddev_v = _mm256_set1_ps(stddev);

  // First fill the data with the uniform numbers. Box-Mueller is a 2 -> 2
  // mapping of 2 uniform numbers to 2 normal numbers (per iteration), so we
  // we need exactly as much space for uniform and normal numbers and can just
  // use the single buffer for both.
  for (int64_t i = 0; i < size; ++i) {
    data[i] = THRandom_uniformFloat(generator, 0, 1);
  }

  for (int64_t i = 0; i < size - 15; i += 16) {
    normal_fill_16_AVX2(data + i, &two_pi, &one, &minus_two, &mean_v, &stddev_v);
  }

  if (size % 16 != 0) {
    // We rewind so that we have 16 values and then compute them in one step.
    data = data + size - 16;
    for (int i = 0; i < 16; ++i) {
      data[i] = THRandom_uniformFloat(generator, 0, 1);
    }
    normal_fill_16_AVX2(data, &two_pi, &one, &minus_two, &mean_v, &stddev_v);
  }
}

void THFloatVector_sigmoid_AVX2(float *y, const float *x, const ptrdiff_t n) {
  ptrdiff_t i;
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 zero = _mm256_set1_ps(0.0f);
  __m256 YMM0, YMM1, YMM2, YMM3;
  for (i = 0; i <= ((n)-16); i += 16) {
    YMM0 = _mm256_loadu_ps(x + i);
    YMM1 = _mm256_loadu_ps(x + i + 8);
    YMM0 = _mm256_sub_ps(zero, YMM0);
    YMM1 = _mm256_sub_ps(zero, YMM1);
    YMM2 = _mm256_add_ps(one, exp256_ps(YMM0));
    YMM3 = _mm256_add_ps(one, exp256_ps(YMM1));
    YMM2 = _mm256_div_ps(one, YMM2);
    YMM3 = _mm256_div_ps(one, YMM3);
    _mm256_storeu_ps(y + i, YMM2);
    _mm256_storeu_ps(y + i + 8, YMM3);
  }
  for (; i < (n); i++) {
    y[i] = 1.0f / (1.0f + expf(-x[i]));
  }
}

#endif // defined(__AVX2__)
