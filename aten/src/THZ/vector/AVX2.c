#if defined(__AVX2__)
#ifndef _MSC_VER
#include <x86intrin.h>
#else
#include <intrin.h>
#endif
#include "AVX2.h"

void THZDoubleVector_cdiv_AVX2(double _Complex *z, const double _Complex *x, const double _Complex *y, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t len = 2 * n;
  ptrdiff_t off;
  double *mx = (double *)x;
  double *my = (double *)y;
  double *mz = (double *)z;

  __m256d YMM0, YMM1, YMM2, YMM3, YMM4, YMM5, YMM6;
  for (i=0; i<=((len)-8); i+=8) {
    // load four complex number from y
    YMM0 = _mm256_loadu_pd(my+i);
    YMM1 = _mm256_loadu_pd(my+i+4);
    // load four complex number from x
    YMM2 = _mm256_loadu_pd(mx+i);
    YMM3 = _mm256_loadu_pd(mx+i+4);

    // YMM4 contains four norms
    // YMM4[63:0]     norm[0]
    // YMM4[127:64]   norm[2]
    // YMM4[191:128]  norm[1]
    // YMM4[255:192]  norm[3]
    YMM4 = _mm256_hadd_pd(_mm256_mul_pd(YMM0, YMM0),
       _mm256_mul_pd(YMM1, YMM1));

    // cal real and imag
    // YMM5 contains four real parts
    // YMM6 contains four imag parts
    // YMM5[63:0]     real[0]  YMM6[63:0]     imag[0]
    // YMM5[127:64]   real[2]  YMM6[127:64]   imag[2]
    // YMM5[191:128]  real[1]  YMM6[191:128]  imag[1]
    // YMM5[255:192]  real[3]  YMM6[255:192]  imag[3]
    YMM5 = _mm256_hadd_pd(_mm256_mul_pd(YMM2, YMM0),
      _mm256_mul_pd(YMM3, YMM1));

    // exchange real and imag in x
    YMM2 = _mm256_shuffle_pd(YMM2, YMM2, 5);
    YMM3 = _mm256_shuffle_pd(YMM3, YMM3, 5);
    YMM6 = _mm256_hsub_pd(_mm256_mul_pd(YMM2, YMM0),
      _mm256_mul_pd(YMM3, YMM1));

    // div by norm
    YMM5 = _mm256_div_pd(YMM5, YMM4);
    YMM6 = _mm256_div_pd(YMM6, YMM4);

    // make them in order
    YMM5 = _mm256_permute4x64_pd(YMM5, 216);
    YMM6 = _mm256_permute4x64_pd(YMM6, 216);

    MAKE_COMPLEX128(YMM0, YMM1, YMM5, YMM6);

    _mm256_storeu_pd(mz+i, YMM0);
    _mm256_storeu_pd(mz+i+4, YMM1);
  }

  off = (n) - ((n)%4);
  for (i=0; i<((n)%4); i++) {
    z[off+i] = x[off+i] / y[off+i];
  }
}

void THZDoubleVector_divs_AVX2(double _Complex *y, const double _Complex *x, const double _Complex c, const ptrdiff_t n) {
  ptrdiff_t i, off;
  double *my = (double *)y;
  double *mx = (double *)x;
  double real = creal(c); double imag = cimag(c);
  double norm = real * real + imag * imag;
  ptrdiff_t len = 2 * n;
  __m256d YMM15 = COMPLEX128_TO_M256D(real/norm, imag/norm);
  __m256d YMM14 = COMPLEX128_TO_M256D(-imag/norm, real/norm);

  __m256d YMM0, YMM1, YMM2, YMM3;
  for (i=0; i<=((len)-8); i+=8) {
    // load 4 complex numbers
    YMM0 = _mm256_loadu_pd(mx+i);
    YMM1 = _mm256_loadu_pd(mx+i+4);

    // cal real & imag
    // [63:0]     real[0]   imag[0]
    // [127:64]   real[2]   imag[2]
    // [191:128]  real[1]   imag[1]
    // [255:192]  real[3]   imag[3]
    HADD_MUL_INORDER_PD(YMM2, YMM0, YMM1, YMM15, YMM15);
    HADD_MUL_INORDER_PD(YMM3, YMM0, YMM1, YMM14, YMM14);

    MAKE_COMPLEX128(YMM0, YMM1, YMM2, YMM3);

    _mm256_storeu_pd(my+i, YMM0);
    _mm256_storeu_pd(my+i+4, YMM1);
  }

  off = (n) - ((n)%4);

  for (i=0; i<((n)%4); i++) {
    y[off+i] = x[off+i] / c;
  }
}

void THZDoubleVector_cmul_AVX2(double _Complex *z, const double _Complex *x, const double _Complex *y, const ptrdiff_t n) {
  ptrdiff_t i, off;
  ptrdiff_t len = 2 * n;

  double *mx = (double *)x;
  double *my = (double *)y;
  double *mz = (double *)z;
  __m256d YMM0, YMM1, YMM2, YMM3, YMM4, YMM5;
  for(i=0;i<=((len)-8); i+=8) {
    // load 4 complex numbers
    YMM0 = _mm256_loadu_pd(mx+i);
    YMM1 = _mm256_loadu_pd(mx+i+4);
    YMM2 = _mm256_loadu_pd(my+i);
    YMM3 = _mm256_loadu_pd(my+i+4);

    HSUB_MUL_INORDER_PD(YMM4, YMM0, YMM1, YMM2, YMM3);
    // exchange real and imag in x
    YMM0 = _mm256_shuffle_pd(YMM0, YMM0, 5);
    YMM1 = _mm256_shuffle_pd(YMM1, YMM1, 5);
    HADD_MUL_INORDER_PD(YMM5, YMM0, YMM1, YMM2, YMM3);

    MAKE_COMPLEX128(YMM0, YMM1, YMM4, YMM5);

    _mm256_storeu_pd(mz+i, YMM0);
    _mm256_storeu_pd(mz+i+4, YMM1);
  }

  off = (n) - ((n)%4);
  for (i=0; i<((n)%4); i++) {
    z[off+i] = x[off+i] * y[off+i];
  }
}

void THZDoubleVector_muls_AVX2(double _Complex *y, const double _Complex *x, const double _Complex c, const ptrdiff_t n) {
  ptrdiff_t i, off;

  double *my = (double *)y;
  double *mx = (double *)x;
  double real = creal(c);
  double imag = cimag(c);
  ptrdiff_t len = 2 * n;

  __m256d YMM15 = COMPLEX128_TO_M256D(real, imag);
  __m256d YMM14 = COMPLEX128_TO_M256D(imag, real);
  __m256d YMM0, YMM1, YMM2, YMM3;
  for(i=0; i<=((len)-8); i+=8) {
    // load 4 complex numbers
    YMM0 = _mm256_loadu_pd(mx+i);
    YMM1 = _mm256_loadu_pd(mx+i+4);

    HSUB_MUL_INORDER_PD(YMM2, YMM0, YMM1, YMM15, YMM15);
    HADD_MUL_INORDER_PD(YMM3, YMM0, YMM1, YMM14, YMM14);

    MAKE_COMPLEX128(YMM0, YMM1, YMM2, YMM3);

    _mm256_storeu_pd(my+i, YMM0);
    _mm256_storeu_pd(my+i+4, YMM1);
  }

  off = (n) - ((n)%4);
  for (i=0; i<((n)%4); i++) {
    y[off+i] = x[off+i] * c;
  }
}

void THZDoubleVector_cadd_AVX2(double _Complex *z, const double _Complex *x, const double _Complex *y, const double _Complex c, const ptrdiff_t n) {
  ptrdiff_t i, off;
  ptrdiff_t len = 2 * n;

  double *mz = (double *)z;
  double *mx = (double *)x;
  double *my = (double *)y;

  double real = creal(c);
  double imag = cimag(c);

  __m256d YMM15 = COMPLEX128_TO_M256D(real, imag);
  __m256d YMM14 = COMPLEX128_TO_M256D(imag, real);
  __m256d YMM0, YMM1, YMM2, YMM3, YMM4, YMM5;

  for(i=0; i<=((len)-8); i+=8) {
    YMM0 = _mm256_loadu_pd(my+i);
    YMM1 = _mm256_loadu_pd(my+i+4);

    HSUB_MUL_INORDER_PD(YMM2, YMM0, YMM1, YMM15, YMM15);
    HADD_MUL_INORDER_PD(YMM3, YMM0, YMM1, YMM14, YMM14);

    MAKE_COMPLEX128(YMM0, YMM1, YMM2, YMM3);

    YMM2 = _mm256_add_pd(_mm256_loadu_pd(mx+i), YMM0);
    YMM3 = _mm256_add_pd(_mm256_loadu_pd(mx+i+4), YMM1);
    _mm256_storeu_pd(mz+i, YMM2);
    _mm256_storeu_pd(mz+i+4, YMM3);
  }

  off = (n) - ((n)%4);
  for (i=0; i<((n)%4); i++) {
    z[off+i] = x[off+i] + y[off+i] * c;
  }
}

void THZFloatVector_cdiv_AVX2(float _Complex *z, const float _Complex *x, const float _Complex *y, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t len = 2 * n;
  ptrdiff_t off;
  float *mx = (float *)x;
  float *my = (float *)y;
  float *mz = (float *)z;

  __m256 YMM0, YMM1, YMM2, YMM3, YMM4, YMM5, YMM6;
  __m256d YMM7, YMM8;
  for (i=0; i<=((len)-16); i+=16) {
    // load 8 complex number from y
    YMM0 = _mm256_loadu_ps(my+i);
    YMM1 = _mm256_loadu_ps(my+i+8);
    // load 8 complex number from x
    YMM2 = _mm256_loadu_ps(mx+i);
    YMM3 = _mm256_loadu_ps(mx+i+8);

    // YMM4 contains 8 norms
    // [31:0]     norm[0]
    // [63:32]    norm[1]
    // [95:64]    norm[4]
    // [127:96]   norm[5]
    // [159:128]  norm[2]
    // [191:160]  norm[3]
    // [223:192]  norm[6]
    // [255:224]  norm[7]
    YMM4 = _mm256_hadd_ps(_mm256_mul_ps(YMM0, YMM0),
       _mm256_mul_ps(YMM1, YMM1));

    // cal real and imag
    //            YMM5     YMM6
    // [31:0]     real[0]  imag[0]
    // [63:32]    real[1]  imag[1]
    // [95:64]    real[4]  imag[4]
    // [127:96]   real[5]  imag[5]
    // [159:128]  real[2]  imag[2]
    // [191:160]  real[3]  imag[3]
    // [223:192]  real[6]  imag[6]
    // [255:224]  real[7]  imag[7]
    YMM5 = _mm256_hadd_ps(_mm256_mul_ps(YMM2, YMM0),
      _mm256_mul_ps(YMM3, YMM1));

    // exchange real and imag in x
    YMM2 = _mm256_shuffle_ps(YMM2, YMM2, 177);
    YMM3 = _mm256_shuffle_ps(YMM3, YMM3, 177);
    YMM6 = _mm256_hsub_ps(_mm256_mul_ps(YMM2, YMM0),
      _mm256_mul_ps(YMM3, YMM1));

    // div by norm
    YMM5 = _mm256_div_ps(YMM5, YMM4);
    YMM6 = _mm256_div_ps(YMM6, YMM4);
    // make them in order
    YMM5 = _mm256_castpd_ps(
      _mm256_permute4x64_pd(
        _mm256_castps_pd(YMM5), 216));
    YMM6 = _mm256_castpd_ps(
      _mm256_permute4x64_pd(
        _mm256_castps_pd(YMM6), 216));

    MAKE_COMPLEX64(YMM0, YMM1, YMM5, YMM6);

    _mm256_storeu_ps(mz+i, YMM0);
    _mm256_storeu_ps(mz+i+8, YMM1);
  }

  off = (n) - ((n)%8);
  for (i=0; i<((n)%8); i++) {
    z[off+i] = x[off+i] / y[off+i];
  }
}

void THZFloatVector_divs_AVX2(float _Complex *y, const float _Complex *x, const float _Complex c, const ptrdiff_t n) {
  ptrdiff_t i, off;
  float *my = (float *)y;
  float *mx = (float *)x;
  float real = crealf(c);
  float imag = cimagf(c);
  float norm = real * real + imag * imag;
  ptrdiff_t len = 2 * n;
  __m256 YMM15 = COMPLEX64_TO_M256(real/norm, imag/norm);
  __m256 YMM14 = COMPLEX64_TO_M256(-imag/norm, real/norm);

  __m256 YMM0, YMM1, YMM2, YMM3;
  for (i=0; i<=((len)-16); i+=16) {
    // load 8 complex numbers
    YMM0 = _mm256_loadu_ps(mx+i);
    YMM1 = _mm256_loadu_ps(mx+i+8);

    HADD_MUL_INORDER_PS(YMM2, YMM0, YMM1, YMM15, YMM15);
    HADD_MUL_INORDER_PS(YMM3, YMM0, YMM1, YMM14, YMM14);
    MAKE_COMPLEX64(YMM0, YMM1, YMM2, YMM3);

    _mm256_storeu_ps(my+i, YMM0);
    _mm256_storeu_ps(my+i+8, YMM1);
  }

  off = (n) - ((n)%8);
  for (i=0; i<((n)%8); i++) {
    y[off+i] = x[off+i] / c;
  }
}

void THZFloatVector_cmul_AVX2(float _Complex *z, const float _Complex *x, const float _Complex *y, const ptrdiff_t n) {
  ptrdiff_t i, off;
  ptrdiff_t len = 2 * n;

  float *mx = (float *)x;
  float *my = (float *)y;
  float *mz = (float *)z;
  __m256 YMM0, YMM1, YMM2, YMM3, YMM4, YMM5;
  for(i=0;i<=((len)-16); i+=16) {
    // load 8 complex numbers
    YMM0 = _mm256_loadu_ps(mx+i);
    YMM1 = _mm256_loadu_ps(mx+i+8);
    YMM2 = _mm256_loadu_ps(my+i);
    YMM3 = _mm256_loadu_ps(my+i+8);

    HSUB_MUL_INORDER_PS(YMM4, YMM0, YMM1, YMM2, YMM3);
    // exchange real and imag in x
    YMM0 = _mm256_shuffle_ps(YMM0, YMM0, 177);
    YMM1 = _mm256_shuffle_ps(YMM1, YMM1, 177);
    HADD_MUL_INORDER_PS(YMM5, YMM0, YMM1, YMM2, YMM3);

    MAKE_COMPLEX64(YMM0, YMM1, YMM4, YMM5);

    _mm256_storeu_ps(mz+i, YMM0);
    _mm256_storeu_ps(mz+i+8, YMM1);
  }

  off = (n) - ((n)%8);
  for (i=0; i<((n)%8); i++) {
    z[off+i] = x[off+i] * y[off+i];
  }
}

void THZFloatVector_muls_AVX2(float _Complex *y, const float _Complex *x, const float _Complex c, const ptrdiff_t n) {
  ptrdiff_t i, off;
  ptrdiff_t len = 2 * n;

  float *my = (float *)y;
  float *mx = (float *)x;
  float real = crealf(c);
  float imag = cimagf(c);

  __m256 YMM15 = COMPLEX64_TO_M256(real, imag);
  __m256 YMM14 = COMPLEX64_TO_M256(imag, real);

  __m256 YMM0,YMM1,YMM2,YMM3;
  for(i=0; i<=((len)-16); i+=16) {
    YMM0 = _mm256_loadu_ps(mx+i);
    YMM1 = _mm256_loadu_ps(mx+i+8);

    HSUB_MUL_INORDER_PS(YMM2, YMM0, YMM1, YMM15, YMM15);
    HADD_MUL_INORDER_PS(YMM3, YMM0, YMM1, YMM14, YMM14);

    MAKE_COMPLEX64(YMM0, YMM1, YMM2, YMM3);

    _mm256_storeu_ps(my+i, YMM0);
    _mm256_storeu_ps(my+i+8, YMM1);
  }

  off = (n) - ((n)%8);
  for (i=0; i<((n)%8); i++) {
    y[off+i] = x[off+i] * c;
  }
}

void THZFloatVector_cadd_AVX2(float _Complex *z, const float _Complex *x, const float _Complex *y, const float _Complex c, const ptrdiff_t n) {
  ptrdiff_t i, off;
  ptrdiff_t len = 2 * n;

  float *mz = (float *)z;
  float *mx = (float *)x;
  float *my = (float *)y;

  float real = crealf(c);
  float imag = cimagf(c);

  __m256 YMM15 = COMPLEX64_TO_M256(real, imag);
  __m256 YMM14 = COMPLEX64_TO_M256(imag, real);

  __m256 YMM0, YMM1, YMM2, YMM3;
  for(i=0;i<=((len)-16); i+=16) {
    YMM0 = _mm256_loadu_ps(my+i);
    YMM1 = _mm256_loadu_ps(my+i+8);

    HSUB_MUL_INORDER_PS(YMM2, YMM0, YMM1, YMM15, YMM15);
    HADD_MUL_INORDER_PS(YMM3, YMM0, YMM1, YMM14, YMM14);

    MAKE_COMPLEX64(YMM0, YMM1, YMM2, YMM3);

    YMM2 = _mm256_add_ps(_mm256_loadu_ps(mx+i), YMM0);
    YMM3 = _mm256_add_ps(_mm256_loadu_ps(mx+i+8), YMM1);
    _mm256_storeu_ps(mz+i, YMM2);
    _mm256_storeu_ps(mz+i+8, YMM3);
  }

  off = (n) - ((n)%8);
  for (i=0; i<((n)%8); i++) {
    z[off+i] = x[off+i] + y[off+i] *c;
  }
}

#endif // defined(__AVX2__)
