#ifndef _MSC_VER
#include <x86intrin.h>
#else
#include <intrin.h>
#endif

static void THDoubleVector_fill_AVX(double *x, const double c, const ptrdiff_t n) {
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

static void THDoubleVector_cdiv_AVX(double *z, const double *x, const double *y, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256d YMM0, YMM1, YMM2, YMM3;
  for (i=0; i<=((n)-8); i+=8) {
    YMM0 = _mm256_loadu_pd(x+i);
    YMM1 = _mm256_loadu_pd(x+i+4);
    YMM2 = _mm256_loadu_pd(y+i);
    YMM3 = _mm256_loadu_pd(y+i+4);
    YMM2 = _mm256_div_pd(YMM0, YMM2);
    YMM3 = _mm256_div_pd(YMM1, YMM3);
    _mm256_storeu_pd(z+i, YMM2);
    _mm256_storeu_pd(z+i+4, YMM3);
  }
  for (; i<(n); i++) {
    z[i] = x[i] / y[i];
  }
}

static void THDoubleVector_div_AVX(double *y, const double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256d YMM15 = _mm256_set_pd(c, c, c, c);
  __m256d YMM0, YMM1, YMM2, YMM3, YMM4, YMM5;
  for (i=0; i<=((n)-8); i+=8) {
    YMM0 = _mm256_loadu_pd(x+i);
    YMM1 = _mm256_loadu_pd(x+i+4);
    YMM4 = _mm256_div_pd(YMM0, YMM15);
    YMM5 = _mm256_div_pd(YMM1, YMM15);
    _mm256_storeu_pd(y+i, YMM4);
    _mm256_storeu_pd(y+i+4, YMM5);
  }
  for (; i<(n); i++) {
    y[i] = x[i] / c;
  }
}

static void THDoubleVector_cmul_AVX(double *z, const double *x, const double *y, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256d YMM0, YMM1, YMM2, YMM3;
  for (i=0; i<=((n)-8); i+=8) {
    YMM0 = _mm256_loadu_pd(x+i);
    YMM1 = _mm256_loadu_pd(x+i+4);
    YMM2 = _mm256_loadu_pd(y+i);
    YMM3 = _mm256_loadu_pd(y+i+4);
    YMM2 = _mm256_mul_pd(YMM0, YMM2);
    YMM3 = _mm256_mul_pd(YMM1, YMM3);
    _mm256_storeu_pd(z+i, YMM2);
    _mm256_storeu_pd(z+i+4, YMM3);
  }
  for (; i<n; i++) {
    z[i] = x[i] * y[i];
  }
}

static void THDoubleVector_mul_AVX(double *y, const double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256d YMM15 = _mm256_set_pd(c, c, c, c);
  __m256d YMM0, YMM1, YMM2, YMM3, YMM4, YMM5;
  for (i=0; i<=((n)-8); i+=8) {
    YMM0 = _mm256_loadu_pd(x+i);
    YMM1 = _mm256_loadu_pd(x+i+4);
    YMM4 = _mm256_mul_pd(YMM0, YMM15);
    YMM5 = _mm256_mul_pd(YMM1, YMM15);
    _mm256_storeu_pd(y+i, YMM4);
    _mm256_storeu_pd(y+i+4, YMM5);
  }
  for (; i<n; i++) {
    y[i] = x[i] * c;
  }
}

static void THDoubleVector_cadd_AVX(double *z, const double *x, const double *y, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256d YMM15 = _mm256_set_pd(c, c, c, c);
  __m256d YMM0, YMM1, YMM2, YMM3, YMM4, YMM5;
  for (i=0; i<=((n)-4); i+=4) {
    YMM0 = _mm256_loadu_pd(y+i);
    YMM1 = _mm256_loadu_pd(x+i);
    YMM2 = _mm256_mul_pd(YMM0, YMM15);
    YMM3 = _mm256_add_pd(YMM1, YMM2);
    _mm256_storeu_pd(z+i, YMM3);
  }
  for (; i<(n); i++) {
    z[i] = x[i] + y[i] * c;
  }
}

static void THDoubleVector_add_AVX(double *y, const double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256d YMM15 = _mm256_set_pd(c, c, c, c);
  __m256d YMM0, YMM1, YMM2, YMM3, YMM4, YMM5;
  for (i=0; i<=((n)-8); i+=8) {
    YMM0 = _mm256_loadu_pd(x+i);
    YMM1 = _mm256_loadu_pd(x+i+4);
    YMM4 = _mm256_add_pd(YMM0, YMM15);
    YMM5 = _mm256_add_pd(YMM1, YMM15);
    _mm256_storeu_pd(y+i, YMM4);
    _mm256_storeu_pd(y+i+4, YMM5);
  }
  for (; i<(n); i++) {
    y[i] = x[i] + c;
  }
}

static void THFloatVector_fill_AVX(float *x, const float c, const ptrdiff_t n) {
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

static void THFloatVector_cdiv_AVX(float *z, const float *x, const float *y, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256 YMM0, YMM1, YMM2, YMM3;
  for (i=0; i<=((n)-16); i+=16) {
    YMM0 = _mm256_loadu_ps(x+i);
    YMM1 = _mm256_loadu_ps(x+i+8);
    YMM2 = _mm256_loadu_ps(y+i);
    YMM3 = _mm256_loadu_ps(y+i+8);
    YMM2 = _mm256_div_ps(YMM0, YMM2);
    YMM3 = _mm256_div_ps(YMM1, YMM3);
    _mm256_storeu_ps(z+i, YMM2);
    _mm256_storeu_ps(z+i+8, YMM3);
  }
  for (; i<(n); i++) {
    z[i] = x[i] / y[i];
  }
}

static void THFloatVector_div_AVX(float *y, const float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256 YMM15 = _mm256_set_ps(c, c, c, c, c, c, c, c);
  __m256 YMM0, YMM1, YMM2, YMM3, YMM4, YMM5;
  for (i=0; i<=((n)-16); i+=16) {
    YMM0 = _mm256_loadu_ps(x+i);
    YMM1 = _mm256_loadu_ps(x+i+8);
    YMM4 = _mm256_div_ps(YMM0, YMM15);
    YMM5 = _mm256_div_ps(YMM1, YMM15);
    _mm256_storeu_ps(y+i, YMM4);
    _mm256_storeu_ps(y+i+8, YMM5);
  }
  for (; i<(n); i++) {
    y[i] = x[i] / c;
  }
}

static void THFloatVector_cmul_AVX(float *z, const float *x, const float *y, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256 YMM0, YMM1, YMM2, YMM3;
  for (i=0; i<=((n)-16); i+=16) {
    YMM0 = _mm256_loadu_ps(x+i);
    YMM1 = _mm256_loadu_ps(x+i+8);
    YMM2 = _mm256_loadu_ps(y+i);
    YMM3 = _mm256_loadu_ps(y+i+8);
    YMM2 = _mm256_mul_ps(YMM0, YMM2);
    YMM3 = _mm256_mul_ps(YMM1, YMM3);
    _mm256_storeu_ps(z+i, YMM2);
    _mm256_storeu_ps(z+i+8, YMM3);
  }
  for (; i<n; i++) {
    z[i] = x[i] * y[i];
  }
}

static void THFloatVector_mul_AVX(float *y, const float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256 YMM15 = _mm256_set_ps(c, c, c, c, c, c, c, c);
  __m256 YMM0, YMM1, YMM2, YMM3, YMM4, YMM5;
  for (i=0; i<=((n)-16); i+=16) {
    YMM0 = _mm256_loadu_ps(x+i);
    YMM1 = _mm256_loadu_ps(x+i+8);
    YMM4 = _mm256_mul_ps(YMM0, YMM15);
    YMM5 = _mm256_mul_ps(YMM1, YMM15);
    _mm256_storeu_ps(y+i, YMM4);
    _mm256_storeu_ps(y+i+8, YMM5);
  }
  for (; i<n; i++) {
    y[i] = x[i] * c;
  }
}

static void THFloatVector_cadd_AVX(float *z, const float *x, const float *y, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256 YMM15 = _mm256_set_ps(c, c, c, c, c, c, c, c);
  __m256 YMM0, YMM1, YMM2, YMM3;
  for (i=0; i<=((n)-8); i+=8) {
    YMM0 = _mm256_loadu_ps(y+i);
    YMM1 = _mm256_loadu_ps(x+i);
    YMM2 = _mm256_mul_ps(YMM0, YMM15);
    YMM3 = _mm256_add_ps(YMM1, YMM2);
    _mm256_storeu_ps(z+i, YMM3);
  }
  for (; i<(n); i++) {
    z[i] = x[i] + y[i] * c;
  }
}

static void THFloatVector_add_AVX(float *y, const float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256 YMM15 = _mm256_set_ps(c, c, c, c, c, c, c, c);
  __m256 YMM0, YMM1, YMM2, YMM3, YMM4, YMM5;
  for (i=0; i<=((n)-16); i+=16) {
    YMM0 = _mm256_loadu_ps(x+i);
    YMM1 = _mm256_loadu_ps(x+i+8);
    YMM4 = _mm256_add_ps(YMM0, YMM15);
    YMM5 = _mm256_add_ps(YMM1, YMM15);
    _mm256_storeu_ps(y+i, YMM4);
    _mm256_storeu_ps(y+i+8, YMM5);
  }
  for (; i<(n); i++) {
    y[i] = x[i] + c;
  }
}
