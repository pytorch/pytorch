#ifndef _MSC_VER
#include <x86intrin.h>
#else
#include <intrin.h>
#endif

static void THDoubleVector_fill_SSE(double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  ptrdiff_t off;
  __m128d XMM0 = _mm_set1_pd(c);
  for (i=0; i<=((n)-8); i+=8) {
    _mm_storeu_pd((x)+i  , XMM0);
    _mm_storeu_pd((x)+i+2, XMM0);
    _mm_storeu_pd((x)+i+4, XMM0);
    _mm_storeu_pd((x)+i+6, XMM0);
  }
  off = (n) - ((n)%8);
  for (i=0; i<((n)%8); i++) {
    x[off+i] = c;
  }
}

static void THDoubleVector_cadd_SSE(double *z, const double *x, const double *y, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m128d XMM7 = _mm_set1_pd(c);
  __m128d XMM0, XMM2;
  for (i=0; i<=((n)-2); i+=2) {
    XMM0 = _mm_loadu_pd((x)+i);
    XMM2 = _mm_loadu_pd((y)+i);
    XMM2 = _mm_mul_pd(XMM2, XMM7);
    XMM2 = _mm_add_pd(XMM0, XMM2);
    _mm_storeu_pd((z)+i, XMM2);
  }
  for (; i<(n); i++) {
    z[i] = x[i] + c * y[i];
  }
}

static void THDoubleVector_adds_SSE(double *y, const double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m128d XMM7 = _mm_set1_pd(c);
  __m128d XMM0, XMM2;
  for (i=0; i<=((n)-4); i+=4) {
    XMM0 = _mm_loadu_pd((x)+i);
    XMM2 = _mm_loadu_pd((x)+i+2);
    XMM0 = _mm_add_pd(XMM0, XMM7);
    XMM2 = _mm_add_pd(XMM2, XMM7);
    _mm_storeu_pd((y)+i, XMM0);
    _mm_storeu_pd((y)+i+2, XMM2);
  }
  for (; i<(n); i++) {
    y[i] = x[i] + c;
  }
}

static void THDoubleVector_cmul_SSE(double *z, const double *x, const double *y, const ptrdiff_t n) {
  ptrdiff_t i;
  for (i=0; i<=((n)-8); i+=8) {
    __m128d XMM0 = _mm_loadu_pd((x)+i  );
    __m128d XMM1 = _mm_loadu_pd((x)+i+2);
    __m128d XMM2 = _mm_loadu_pd((x)+i+4);
    __m128d XMM3 = _mm_loadu_pd((x)+i+6);
    __m128d XMM4 = _mm_loadu_pd((y)+i  );
    __m128d XMM5 = _mm_loadu_pd((y)+i+2);
    __m128d XMM6 = _mm_loadu_pd((y)+i+4);
    __m128d XMM7 = _mm_loadu_pd((y)+i+6);
    XMM4 = _mm_mul_pd(XMM4, XMM0);
    XMM5 = _mm_mul_pd(XMM5, XMM1);
    XMM6 = _mm_mul_pd(XMM6, XMM2);
    XMM7 = _mm_mul_pd(XMM7, XMM3);
    _mm_storeu_pd((z)+i  , XMM4);
    _mm_storeu_pd((z)+i+2, XMM5);
    _mm_storeu_pd((z)+i+4, XMM6);
    _mm_storeu_pd((z)+i+6, XMM7);
  }
  for (; i<(n); i++) {
    z[i] = x[i] * y[i];
  }
}

static void THDoubleVector_muls_SSE(double *y, const double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m128d XMM15 = _mm_set1_pd(c);
  for (i=0; i<=((n)-8); i+=8) {
    __m128d XMM0 = _mm_loadu_pd((x)+i  );
    __m128d XMM1 = _mm_loadu_pd((x)+i+2);
    __m128d XMM2 = _mm_loadu_pd((x)+i+4);
    __m128d XMM3 = _mm_loadu_pd((x)+i+6);
    __m128d XMM4 = _mm_mul_pd(XMM15, XMM0);
    __m128d XMM5 = _mm_mul_pd(XMM15, XMM1);
    __m128d XMM6 = _mm_mul_pd(XMM15, XMM2);
    __m128d XMM7 = _mm_mul_pd(XMM15, XMM3);
    _mm_storeu_pd((y)+i  , XMM4);
    _mm_storeu_pd((y)+i+2, XMM5);
    _mm_storeu_pd((y)+i+4, XMM6);
    _mm_storeu_pd((y)+i+6, XMM7);
  }
  for (; i<(n); i++) {
    y[i] = x[i] * c;
  }
}

static void THDoubleVector_cdiv_SSE(double *z, const double *x, const double *y, const ptrdiff_t n) {
  ptrdiff_t i;
  __m128d XMM0, XMM1, XMM2, XMM3;
  for (i=0; i<=((n)-4); i+=4) {
    XMM0 = _mm_loadu_pd(x+i);
    XMM1 = _mm_loadu_pd(x+i+2);
    XMM2 = _mm_loadu_pd(y+i);
    XMM3 = _mm_loadu_pd(y+i+2);
    XMM2 = _mm_div_pd(XMM0, XMM2);
    XMM3 = _mm_div_pd(XMM1, XMM3);
    _mm_storeu_pd(z+i, XMM2);
    _mm_storeu_pd(z+i+2, XMM3);
  }
  for (; i<(n); i++) {
    z[i] = x[i] / y[i];
  }
}

static void THDoubleVector_divs_SSE(double *y, const double *x, const double c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m128d XMM7 = _mm_set1_pd(c);
  __m128d XMM0, XMM1;
  for (i=0; i<=((n)-4); i+=4) {
    XMM0 = _mm_loadu_pd(x+i);
    XMM1 = _mm_loadu_pd(x+i+2);
    XMM0 = _mm_div_pd(XMM0, XMM7);
    XMM1 = _mm_div_pd(XMM1, XMM7);
    _mm_storeu_pd(y+i, XMM0);
    _mm_storeu_pd(y+i+2, XMM1);
  }
  for (; i<(n); i++) {
    y[i] = x[i] / c;
  }
}

static void THFloatVector_fill_SSE(float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m128 XMM0 = _mm_set_ps1(c);
  ptrdiff_t off;
  for (i=0; i<=((n)-16); i+=16) {
    _mm_storeu_ps((x)+i  ,  XMM0);
    _mm_storeu_ps((x)+i+4,  XMM0);
    _mm_storeu_ps((x)+i+8,  XMM0);
    _mm_storeu_ps((x)+i+12, XMM0);
  }
  off = (n) - ((n)%16);
  for (i=0; i<((n)%16); i++) {
    x[off+i] = c;
  }
}


static void THFloatVector_cadd_SSE(float *z, const float *x, const float *y, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m128 XMM7 = _mm_set_ps1(c);
  __m128 XMM0, XMM2;
  for (i=0; i<=((n)-4); i+=4) {
    XMM0 = _mm_loadu_ps((x)+i);
    XMM2 = _mm_loadu_ps((y)+i);
    XMM2 = _mm_mul_ps(XMM2, XMM7);
    XMM2 = _mm_add_ps(XMM0, XMM2);
    _mm_storeu_ps((z)+i, XMM2);
  }
  for (; i<(n); i++) {
    z[i] = x[i] + c * y[i];
  }
}

static void THFloatVector_adds_SSE(float *y, const float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m128 XMM7 = _mm_set1_ps(c);
  __m128 XMM0, XMM2;
  for (i=0; i<=((n)-8); i+=8) {
    XMM0 = _mm_loadu_ps((x)+i);
    XMM2 = _mm_loadu_ps((x)+i+4);
    XMM0 = _mm_add_ps(XMM0, XMM7);
    XMM2 = _mm_add_ps(XMM2, XMM7);
    _mm_storeu_ps((y)+i, XMM0);
    _mm_storeu_ps((y)+i+4, XMM2);
  }
  for (; i<(n); i++) {
    y[i] = x[i] + c;
  }
}

static void THFloatVector_cmul_SSE(float *z, const float *x, const float *y, const ptrdiff_t n) {
  ptrdiff_t i;
  for (i=0; i<=((n)-16); i+=16) {
    __m128 XMM0 = _mm_loadu_ps((x)+i   );
    __m128 XMM1 = _mm_loadu_ps((x)+i+ 4);
    __m128 XMM2 = _mm_loadu_ps((x)+i+ 8);
    __m128 XMM3 = _mm_loadu_ps((x)+i+12);
    __m128 XMM4 = _mm_loadu_ps((y)+i   );
    __m128 XMM5 = _mm_loadu_ps((y)+i+ 4);
    __m128 XMM6 = _mm_loadu_ps((y)+i+ 8);
    __m128 XMM7 = _mm_loadu_ps((y)+i+12);
    XMM4 = _mm_mul_ps(XMM4, XMM0);
    XMM5 = _mm_mul_ps(XMM5, XMM1);
    XMM6 = _mm_mul_ps(XMM6, XMM2);
    XMM7 = _mm_mul_ps(XMM7, XMM3);
    _mm_storeu_ps((z)+i   , XMM4);
    _mm_storeu_ps((z)+i+ 4, XMM5);
    _mm_storeu_ps((z)+i+ 8, XMM6);
    _mm_storeu_ps((z)+i+12, XMM7);
  }
  for (; i<(n); i++) {
    z[i] = x[i] * y[i];
  }
}

static void THFloatVector_muls_SSE(float *y, const float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m128 XMM15 = _mm_set_ps1(c);
  for (i=0; i<=((n)-16); i+=16) {
    __m128 XMM0 = _mm_loadu_ps((x)+i   );
    __m128 XMM1 = _mm_loadu_ps((x)+i+ 4);
    __m128 XMM2 = _mm_loadu_ps((x)+i+ 8);
    __m128 XMM3 = _mm_loadu_ps((x)+i+12);
    __m128 XMM4 = _mm_mul_ps(XMM15, XMM0);
    __m128 XMM5 = _mm_mul_ps(XMM15, XMM1);
    __m128 XMM6 = _mm_mul_ps(XMM15, XMM2);
    __m128 XMM7 = _mm_mul_ps(XMM15, XMM3);
    _mm_storeu_ps((y)+i   , XMM4);
    _mm_storeu_ps((y)+i+ 4, XMM5);
    _mm_storeu_ps((y)+i+ 8, XMM6);
    _mm_storeu_ps((y)+i+12, XMM7);
  }
  for (; i<(n); i++) {
    y[i] = x[i] * c;
  }
}

static void THFloatVector_cdiv_SSE(float *z, const float *x, const float *y, const ptrdiff_t n) {
  ptrdiff_t i;
  __m128 XMM0, XMM1, XMM2, XMM3;
  for (i=0; i<=((n)-8); i+=8) {
    XMM0 = _mm_loadu_ps(x+i);
    XMM1 = _mm_loadu_ps(x+i+4);
    XMM2 = _mm_loadu_ps(y+i);
    XMM3 = _mm_loadu_ps(y+i+4);
    XMM2 = _mm_div_ps(XMM0, XMM2);
    XMM3 = _mm_div_ps(XMM1, XMM3);
    _mm_storeu_ps(z+i, XMM2);
    _mm_storeu_ps(z+i+4, XMM3);
  }
  for (; i<(n); i++) {
    z[i] = x[i] / y[i];
  }
}

static void THFloatVector_divs_SSE(float *y, const float *x, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m128 XMM7 = _mm_set1_ps(c);
  __m128 XMM0, XMM1;
  for (i=0; i<=((n)-8); i+=8) {
    XMM0 = _mm_loadu_ps(x+i);
    XMM1 = _mm_loadu_ps(x+i+4);
    XMM0 = _mm_div_ps(XMM0, XMM7);
    XMM1 = _mm_div_ps(XMM1, XMM7);
    _mm_storeu_ps(y+i, XMM0);
    _mm_storeu_ps(y+i+4, XMM1);
  }
  for (; i<(n); i++) {
    y[i] = x[i] / c;
  }
}
