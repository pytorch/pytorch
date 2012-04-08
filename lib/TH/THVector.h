#ifndef TH_VECTOR_INC
#define TH_VECTOR_INC

#include "THGeneral.h"

#define THVector_(NAME) TH_CONCAT_4(TH,Real,Vector_,NAME)

#if defined USE_SSE2 || defined USE_SSE3 || defined USE_SSSE3 \
  || defined USE_SSE4_1 || defined USE_SSE4_2

#ifdef USE_SSE2
#include <emmintrin.h>
#endif

#ifdef USE_SSE3
#include <pmmintrin.h>
#endif

#ifdef USE_SSSE3
#include <tmmintrin.h>
#endif

#if defined (USE_SSE4_2) || defined (USE_SSE4_1)
#include <smmintrin.h>
#endif

#define THDoubleVector_fill(x, c, n) {          \
    long i;                                     \
    __m128d XMM0 = _mm_set1_pd(c);              \
    for (i=0; i<=((n)-8); i+=8) {               \
      _mm_storeu_pd((x)+i  , XMM0);             \
      _mm_storeu_pd((x)+i+2, XMM0);             \
      _mm_storeu_pd((x)+i+4, XMM0);             \
      _mm_storeu_pd((x)+i+6, XMM0);             \
    }                                           \
    long off = (n) - ((n)%8);                   \
    for (i=0; i<((n)%8); i++) {                 \
      x[off+i] = c;                             \
    }                                           \
  }


#define THDoubleVector_add(y, x, c, n) {        \
        long i = 0;                             \
        __m128d XMM7 = _mm_set1_pd(c);          \
        __m128d XMM0,XMM1,XMM2;                 \
        __m128d XMM3,XMM4,XMM5;                 \
        for (; i<=((n)-6); i+=6) {              \
            XMM0 = _mm_loadu_pd((x)+i);         \
            XMM1 = _mm_loadu_pd((x)+i+2);       \
            XMM2 = _mm_loadu_pd((x)+i+4);       \
            XMM3 = _mm_loadu_pd((y)+i);         \
            XMM4 = _mm_loadu_pd((y)+i+2);       \
            XMM5 = _mm_loadu_pd((y)+i+4);       \
            XMM0 = _mm_mul_pd(XMM0, XMM7);      \
            XMM1 = _mm_mul_pd(XMM1, XMM7);      \
            XMM2 = _mm_mul_pd(XMM2, XMM7);      \
            XMM3 = _mm_add_pd(XMM3, XMM0);      \
            XMM4 = _mm_add_pd(XMM4, XMM1);      \
            XMM5 = _mm_add_pd(XMM5, XMM2);      \
            _mm_storeu_pd((y)+i  , XMM3);       \
            _mm_storeu_pd((y)+i+2, XMM4);       \
            _mm_storeu_pd((y)+i+4, XMM5);       \
        }                                       \
        for (; i<(n); i++) {                    \
            y[i] += c * x[i];                   \
        }                                       \
    }

#define THDoubleVector_diff(z, x, y, n) {       \
    long i;                                     \
    for (i=0; i<=((n)-8); i+=8) {               \
      __m128d XMM0 = _mm_loadu_pd((x)+i  );     \
      __m128d XMM1 = _mm_loadu_pd((x)+i+2);     \
      __m128d XMM2 = _mm_loadu_pd((x)+i+4);     \
      __m128d XMM3 = _mm_loadu_pd((x)+i+6);     \
      __m128d XMM4 = _mm_loadu_pd((y)+i  );     \
      __m128d XMM5 = _mm_loadu_pd((y)+i+2);     \
      __m128d XMM6 = _mm_loadu_pd((y)+i+4);     \
      __m128d XMM7 = _mm_loadu_pd((y)+i+6);     \
      XMM0 = _mm_sub_pd(XMM0, XMM4);            \
      XMM1 = _mm_sub_pd(XMM1, XMM5);            \
      XMM2 = _mm_sub_pd(XMM2, XMM6);            \
      XMM3 = _mm_sub_pd(XMM3, XMM7);            \
      _mm_storeu_pd((z)+i  , XMM0);             \
      _mm_storeu_pd((z)+i+2, XMM1);             \
      _mm_storeu_pd((z)+i+4, XMM2);             \
      _mm_storeu_pd((z)+i+6, XMM3);             \
    }                                           \
    long off = (n) - ((n)%8);                   \
    for (i=0; i<((n)%8); i++) {                 \
      z[off+i] = x[off+i] - y[off+i];           \
    }                                           \
  }

#define THDoubleVector_scale(y, c, n) {         \
    long i;                                     \
    __m128d XMM7 = _mm_set1_pd(c);              \
    for (i=0; i<=((n)-4); i+=4) {               \
      __m128d XMM0 = _mm_loadu_pd((y)+i  );     \
      __m128d XMM1 = _mm_loadu_pd((y)+i+2);     \
      XMM0 = _mm_mul_pd(XMM0, XMM7);            \
      XMM1 = _mm_mul_pd(XMM1, XMM7);            \
      _mm_storeu_pd((y)+i  , XMM0);             \
      _mm_storeu_pd((y)+i+2, XMM1);             \
    }                                           \
    long off = (n) - ((n)%4);                   \
    for (i=0; i<((n)%4); i++) {                 \
      y[off+i] *= c;                            \
    }                                           \
  }

#define THDoubleVector_mul(y, x, n) {           \
    long i;                                     \
    for (i=0; i<=((n)-8); i+=8) {               \
      __m128d XMM0 = _mm_loadu_pd((x)+i  );     \
      __m128d XMM1 = _mm_loadu_pd((x)+i+2);     \
      __m128d XMM2 = _mm_loadu_pd((x)+i+4);     \
      __m128d XMM3 = _mm_loadu_pd((x)+i+6);     \
      __m128d XMM4 = _mm_loadu_pd((y)+i  );     \
      __m128d XMM5 = _mm_loadu_pd((y)+i+2);     \
      __m128d XMM6 = _mm_loadu_pd((y)+i+4);     \
      __m128d XMM7 = _mm_loadu_pd((y)+i+6);     \
      XMM4 = _mm_mul_pd(XMM4, XMM0);            \
      XMM5 = _mm_mul_pd(XMM5, XMM1);            \
      XMM6 = _mm_mul_pd(XMM6, XMM2);            \
      XMM7 = _mm_mul_pd(XMM7, XMM3);            \
      _mm_storeu_pd((y)+i  , XMM4);             \
      _mm_storeu_pd((y)+i+2, XMM5);             \
      _mm_storeu_pd((y)+i+4, XMM6);             \
      _mm_storeu_pd((y)+i+6, XMM7);             \
    }                                           \
    long off = (n) - ((n)%8);                   \
    for (i=0; i<((n)%8); i++) {                 \
      y[off+i] *= x[off+i];                     \
    }                                           \
  }

#define THFloatVector_fill(x, c, n) {           \
    long i;                                     \
    __m128 XMM0 = _mm_set_ps1(c);               \
    for (i=0; i<=((n)-16); i+=16) {             \
      _mm_storeu_ps((x)+i  ,  XMM0);            \
      _mm_storeu_ps((x)+i+4,  XMM0);            \
      _mm_storeu_ps((x)+i+8,  XMM0);            \
      _mm_storeu_ps((x)+i+12, XMM0);            \
    }                                           \
    long off = (n) - ((n)%16);                  \
    for (i=0; i<((n)%16); i++) {                \
      x[off+i] = c;                             \
    }                                           \
  }

#define THFloatVector_add(y, x, c, n) {         \
        long i = 0;                             \
        __m128 XMM7 = _mm_set_ps1(c);           \
        __m128 XMM0,XMM1,XMM2;                  \
        __m128 XMM3,XMM4,XMM5;                  \
        for (; i<=((n)-12); i+=12) {            \
            XMM0 = _mm_loadu_ps((x)+i);         \
            XMM1 = _mm_loadu_ps((x)+i+4);       \
            XMM2 = _mm_loadu_ps((x)+i+8);       \
            XMM3 = _mm_loadu_ps((y)+i);         \
            XMM4 = _mm_loadu_ps((y)+i+4);       \
            XMM5 = _mm_loadu_ps((y)+i+8);       \
            XMM0 = _mm_mul_ps(XMM0, XMM7);      \
            XMM1 = _mm_mul_ps(XMM1, XMM7);      \
            XMM2 = _mm_mul_ps(XMM2, XMM7);      \
            XMM3 = _mm_add_ps(XMM3, XMM0);      \
            XMM4 = _mm_add_ps(XMM4, XMM1);      \
            XMM5 = _mm_add_ps(XMM5, XMM2);      \
            _mm_storeu_ps((y)+i  , XMM3);       \
            _mm_storeu_ps((y)+i+4, XMM4);       \
            _mm_storeu_ps((y)+i+8, XMM5);       \
        }                                       \
        for (; i<(n); i++) {                    \
            y[i] += c * x[i];                   \
        }                                       \
    }

#define THFloatVector_diff(z, x, y, n) {        \
    long i;                                     \
    for (i=0; i<=((n)-16); i+=16) {             \
      __m128 XMM0 = _mm_loadu_ps((x)+i   );     \
      __m128 XMM1 = _mm_loadu_ps((x)+i+ 4);     \
      __m128 XMM2 = _mm_loadu_ps((x)+i+ 8);     \
      __m128 XMM3 = _mm_loadu_ps((x)+i+12);     \
      __m128 XMM4 = _mm_loadu_ps((y)+i   );     \
      __m128 XMM5 = _mm_loadu_ps((y)+i+ 4);     \
      __m128 XMM6 = _mm_loadu_ps((y)+i+ 8);     \
      __m128 XMM7 = _mm_loadu_ps((y)+i+12);     \
      XMM0 = _mm_sub_ps(XMM0, XMM4);            \
      XMM1 = _mm_sub_ps(XMM1, XMM5);            \
      XMM2 = _mm_sub_ps(XMM2, XMM6);            \
      XMM3 = _mm_sub_ps(XMM3, XMM7);            \
      _mm_storeu_ps((z)+i   , XMM0);            \
      _mm_storeu_ps((z)+i+ 4, XMM1);            \
      _mm_storeu_ps((z)+i+ 8, XMM2);            \
      _mm_storeu_ps((z)+i+12, XMM3);            \
    }                                           \
    long off = (n) - ((n)%16);                  \
    for (i=0; i<((n)%16); i++) {                \
      z[off+i] = x[off+i] - y[off+i];           \
    }                                           \
  }

#define THFloatVector_scale(y, c, n) {          \
    long i;                                     \
    __m128 XMM7 = _mm_set_ps1(c);               \
    for (i=0; i<=((n)-8); i+=8) {               \
      __m128 XMM0 = _mm_loadu_ps((y)+i  );      \
      __m128 XMM1 = _mm_loadu_ps((y)+i+4);      \
      XMM0 = _mm_mul_ps(XMM0, XMM7);            \
      XMM1 = _mm_mul_ps(XMM1, XMM7);            \
      _mm_storeu_ps((y)+i  , XMM0);             \
      _mm_storeu_ps((y)+i+4, XMM1);             \
    }                                           \
    long off = (n) - ((n)%8);                   \
    for (i=0; i<((n)%8); i++) {                 \
      y[off+i] *= c;                            \
    }                                           \
  }

#define THFloatVector_mul(y, x, n) {            \
    long i;                                     \
    for (i=0; i<=((n)-16); i+=16) {             \
      __m128 XMM0 = _mm_loadu_ps((x)+i   );     \
      __m128 XMM1 = _mm_loadu_ps((x)+i+ 4);     \
      __m128 XMM2 = _mm_loadu_ps((x)+i+ 8);     \
      __m128 XMM3 = _mm_loadu_ps((x)+i+12);     \
      __m128 XMM4 = _mm_loadu_ps((y)+i   );     \
      __m128 XMM5 = _mm_loadu_ps((y)+i+ 4);     \
      __m128 XMM6 = _mm_loadu_ps((y)+i+ 8);     \
      __m128 XMM7 = _mm_loadu_ps((y)+i+12);     \
      XMM4 = _mm_mul_ps(XMM4, XMM0);            \
      XMM5 = _mm_mul_ps(XMM5, XMM1);            \
      XMM6 = _mm_mul_ps(XMM6, XMM2);            \
      XMM7 = _mm_mul_ps(XMM7, XMM3);            \
      _mm_storeu_ps((y)+i   , XMM4);            \
      _mm_storeu_ps((y)+i+ 4, XMM5);            \
      _mm_storeu_ps((y)+i+ 8, XMM6);            \
      _mm_storeu_ps((y)+i+12, XMM7);            \
    }                                           \
    long off = (n) - ((n)%16);                  \
    for (i=0; i<((n)%16); i++) {                \
      y[off+i] *= x[off+i];                     \
    }                                           \
  }

#else

/* If SSE2 not defined, then generate plain C operators */
#include "generic/THVector.c"
#include "THGenerateFloatTypes.h"

#endif

/* For non-float types, generate plain C operators */
#include "generic/THVector.c"
#include "THGenerateIntTypes.h"

#endif
