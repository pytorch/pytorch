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
    long off;                                   \
    __m128d XMM0 = _mm_set1_pd(c);              \
    for (i=0; i<=((n)-8); i+=8) {               \
      _mm_storeu_pd((x)+i  , XMM0);             \
      _mm_storeu_pd((x)+i+2, XMM0);             \
      _mm_storeu_pd((x)+i+4, XMM0);             \
      _mm_storeu_pd((x)+i+6, XMM0);             \
    }                                           \
    off = (n) - ((n)%8);                        \
    for (i=0; i<((n)%8); i++) {                 \
      x[off+i] = c;                             \
    }                                           \
  }


#define THDoubleVector_add(y, x, c, n) {        \
    long i = 0;                                 \
    __m128d XMM7 = _mm_set1_pd(c);              \
    __m128d XMM0,XMM2;                          \
    for (; i<=((n)-2); i+=2) {                  \
      XMM0 = _mm_loadu_pd((x)+i);               \
      XMM2 = _mm_loadu_pd((y)+i);               \
      XMM0 = _mm_mul_pd(XMM0, XMM7);            \
      XMM2 = _mm_add_pd(XMM2, XMM0);            \
      _mm_storeu_pd((y)+i  , XMM2);             \
    }                                           \
    for (; i<(n); i++) {                        \
      y[i] += c * x[i];                         \
    }                                           \
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
    long off;                                   \
    for (i=0; i<=((n)-16); i+=16) {             \
      _mm_storeu_ps((x)+i  ,  XMM0);            \
      _mm_storeu_ps((x)+i+4,  XMM0);            \
      _mm_storeu_ps((x)+i+8,  XMM0);            \
      _mm_storeu_ps((x)+i+12, XMM0);            \
    }                                           \
    off = (n) - ((n)%16);                       \
    for (i=0; i<((n)%16); i++) {                \
      x[off+i] = c;                             \
    }                                           \
  }

#define THFloatVector_add(y, x, c, n) {         \
    long i = 0;                                 \
    __m128 XMM7 = _mm_set_ps1(c);               \
    __m128 XMM0,XMM2;                           \
    for (; i<=((n)-4); i+=4) {                  \
      XMM0 = _mm_loadu_ps((x)+i);               \
      XMM2 = _mm_loadu_ps((y)+i);               \
      XMM0 = _mm_mul_ps(XMM0, XMM7);            \
      XMM2 = _mm_add_ps(XMM2, XMM0);            \
      _mm_storeu_ps((y)+i  , XMM2);             \
    }                                           \
    for (; i<(n); i++) {                        \
      y[i] += c * x[i];                         \
    }                                           \
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

#elif defined __NEON__
/* ARM NEON Assembly routine for operating on floats */

#define THFloatVector_fill(x, c, n) {                   \
        float ctemp = c;                                \
        float * caddr = &ctemp;                         \
        __asm__ __volatile__ (                          \
            "mov         r0, %0           @ \n\t"       \
            "ldr         r4, [%1]         @ \n\t"       \
            "vdup.32     q12, r4          @ \n\t"       \
            "vdup.32     q13, r4          @ \n\t"       \
            "lsrs        r4, %2, #3       @ \n\t"       \
            "beq         3f               @ \n\t"       \
            "1:                           @ \n\t"       \
            "vst1.32     {d24-d27}, [r0]! @ \n\t"       \
            "subs        r4, r4, #1       @ \n\t"       \
            "bne         1b               @ \n\t"       \
            "3:                           @ \n\t"       \
            "ands        r4, %2, #7       @ \n\t"       \
            "beq         5f               @ \n\t"       \
            "4:                           @ \n\t"       \
            "subs        r4, r4, #1       @ \n\t"       \
            "vst1.32     {d24[0]}, [r0]!  @ \n\t"       \
            "bne         4b               @ \n\t"       \
            "5:                           @ "           \
            :                                           \
            :"r" (x), "r"(caddr),"r"(n)                 \
            : "cc", "r0", "r4",  "memory",              \
              "q12",                                    \
              "d24", "d25", "d26", "d27"                \
            );                                          \
    }

#define THFloatVector_diff(z, x, y, n) {                                \
        __asm__ __volatile__ (                                          \
            "mov         r0, %2           @ \n\t"                       \
            "mov         r1, %1           @ \n\t"                       \
            "mov         r2, %0           @ \n\t"                       \
            "lsrs        r4, %3, #3       @ \n\t"                       \
            "beq         3f               @ \n\t"                       \
            "vld1.32     {d16-d19}, [r1]! @ \n\t"                       \
            "vld1.32     {d0-d3}, [r0]!   @ \n\t"                       \
            "1:                           @ \n\t"                       \
            "vsub.f32    q12, q8, q0      @ \n\t"                       \
            "vsub.f32    q13, q9, q1      @ \n\t"                       \
            "subs        r4, r4, #1       @ \n\t"                       \
            "beq         2f               @ \n\t"                       \
            "vld1.32     {d16-d19}, [r1]! @ \n\t"                       \
            "vld1.32     {d0-d3}, [r0]!   @ \n\t"                       \
            "vst1.32     {d24-d27}, [r2]! @ \n\t"                       \
            "b           1b               @ \n\t"                       \
            "2:                           @ \n\t"                       \
            "vst1.32     {d24-d27}, [r2]! @ \n\t"                       \
            "3:                           @ \n\t"                       \
            "ands        r4, %3, #7       @ \n\t"                       \
            "beq         5f               @ \n\t"                       \
            "4:                           @ \n\t"                       \
            "subs        r4, r4, #1       @ \n\t"                       \
            "vld1.32     {d16[0]}, [r1]!  @ \n\t"                       \
            "vld1.32     {d0[0]}, [r0]!   @ \n\t"                       \
            "vsub.f32    d24, d16, d0     @ \n\t"                       \
            "vst1.32     {d24[0]}, [r2]!  @ \n\t"                       \
            "bne         4b               @ \n\t"                       \
            "5:                           @ "                           \
            :                                                           \
            :"r" (z), "r" (x),"r" (y), "r"(n)                           \
            : "cc", "r0", "r1", "r2", "r4", "memory",                   \
              "q0", "q1", "q8", "q9", "q12", "q13",                     \
              "d0", "d1", "d2", "d3",                                   \
              "d16", "d17", "d18", "d19", "d24", "d25", "d26", "d27"    \
            );                                                          \
    }

#define THFloatVector_scale(y, c, n) {                                  \
        float ctemp = c;                                                \
        float * caddr = &ctemp;                                         \
        __asm__ __volatile__ (                                          \
            "mov         r0, %0           @ \n\t"                       \
            "mov         r2, r0           @ \n\t"                       \
            "ldr         r5, [%1]         @ \n\t"                       \
            "vdup.32     q14, r5          @ \n\t"                       \
            "lsrs        r5, %2, #5       @ \n\t"                       \
            "beq         3f               @ \n\t"                       \
            "vld1.32     {d0-d3}, [r0]!   @ \n\t"                       \
            "vld1.32     {d4-d7}, [r0]!   @ \n\t"                       \
            "vld1.32     {d8-d11}, [r0]!  @ \n\t"                       \
            "vld1.32     {d12-d15}, [r0]! @ \n\t"                       \
            "1:                           @ \n\t"                       \
            "vmul.f32    q0, q0, q14      @ \n\t"                       \
            "vmul.f32    q1, q1, q14      @ \n\t"                       \
            "vmul.f32    q2, q2, q14      @ \n\t"                       \
            "vmul.f32    q3, q3, q14      @ \n\t"                       \
            "vmul.f32    q4, q4, q14      @ \n\t"                       \
            "vmul.f32    q5, q5, q14      @ \n\t"                       \
            "vmul.f32    q6, q6, q14      @ \n\t"                       \
            "vmul.f32    q7, q7, q14      @ \n\t"                       \
            "subs        r5, r5, #1       @ \n\t"                       \
            "beq         2f               @ \n\t"                       \
            "vst1.32     {d0-d3}, [r2]!   @ \n\t"                       \
            "vld1.32     {d0-d3}, [r0]!   @ \n\t"                       \
            "vst1.32     {d4-d7}, [r2]!   @ \n\t"                       \
            "vld1.32     {d4-d7}, [r0]!   @ \n\t"                       \
            "vst1.32     {d8-d11}, [r2]!  @ \n\t"                       \
            "vld1.32     {d8-d11}, [r0]!  @ \n\t"                       \
            "vst1.32     {d12-d15}, [r2]! @ \n\t"                       \
            "vld1.32     {d12-d15}, [r0]! @ \n\t"                       \
            "b           1b               @ \n\t"                       \
            "2:                           @ \n\t"                       \
            "vst1.32     {d0-d3}, [r2]!   @ \n\t"                       \
            "vst1.32     {d4-d7}, [r2]!   @ \n\t"                       \
            "vst1.32     {d8-d11}, [r2]!  @ \n\t"                       \
            "vst1.32     {d12-d15}, [r2]! @ \n\t"                       \
            "3:                           @ \n\t"                       \
            "lsrs        r5, %2, #4       @ \n\t"                       \
            "ands        r5, r5, #1       @ \n\t"                       \
            "beq         4f               @ \n\t"                       \
            "vld1.32     {d0-d3}, [r0]!   @ \n\t"                       \
            "vld1.32     {d4-d7}, [r0]!   @ \n\t"                       \
            "vmul.f32    q0, q0, q14      @ \n\t"                       \
            "vmul.f32    q1, q1, q14      @ \n\t"                       \
            "vmul.f32    q2, q2, q14      @ \n\t"                       \
            "vmul.f32    q3, q3, q14      @ \n\t"                       \
            "vst1.32     {d0-d3}, [r2]!   @ \n\t"                       \
            "vst1.32     {d4-d7}, [r2]!   @ \n\t"                       \
            "4:                           @ \n\t"                       \
            "lsrs        r5, %2, #3       @ \n\t"                       \
            "ands        r5, r5, #1       @ \n\t"                       \
            "beq         5f               @ \n\t"                       \
            "vld1.32     {d0-d3}, [r0]!   @ \n\t"                       \
            "vmul.f32    q0, q0, q14      @ \n\t"                       \
            "vmul.f32    q1, q1, q14      @ \n\t"                       \
            "vst1.32     {d0-d3}, [r2]!   @ \n\t"                       \
            "5:                           @ \n\t"                       \
            "ands        r5, %2, #7       @ \n\t"                       \
            "beq         7f               @ \n\t"                       \
            "6:                           @ \n\t"                       \
            "subs        r5, r5, #1       @ \n\t"                       \
            "vld1.32     d0[0], [r0]!     @ \n\t"                       \
            "vmul.f32    d0, d0, d28      @ \n\t"                       \
            "vst1.32     d0[0], [r2]!     @ \n\t"                       \
            "bne         6b               @ \n\t"                       \
            "7:                           @ "                           \
            :                                                           \
            :"r" (y), "r"(caddr),"r"(n)                                 \
            : "cc", "r0", "r2", "r5", "memory",                         \
              "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q14",    \
              "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",           \
              "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",     \
              "d28", "d29"                                              \
            );                                                          \
    }

#define THFloatVector_mul(y, x, n) {                                    \
        __asm__ __volatile__ (                                          \
            "mov         r0, %0           @ \n\t"                       \
            "mov         r1, %1           @ \n\t"                       \
            "mov         r2, r0           @ \n\t"                       \
            "lsrs        r4, %2, #3       @ \n\t"                       \
            "beq         3f               @ \n\t"                       \
            "vld1.32     {d16-d19}, [r1]! @ \n\t"                       \
            "vld1.32     {d0-d3}, [r0]!   @ \n\t"                       \
            "1:                           @ \n\t"                       \
            "vmul.f32    q12, q8, q0      @ \n\t"                       \
            "vmul.f32    q13, q9, q1      @ \n\t"                       \
            "subs        r4, r4, #1       @ \n\t"                       \
            "beq         2f               @ \n\t"                       \
            "vld1.32     {d16-d19}, [r1]! @ \n\t"                       \
            "vld1.32     {d0-d3}, [r0]!   @ \n\t"                       \
            "vst1.32     {d24-d27}, [r2]! @ \n\t"                       \
            "b           1b               @ \n\t"                       \
            "2:                           @ \n\t"                       \
            "vst1.32     {d24-d27}, [r2]! @ \n\t"                       \
            "3:                           @ \n\t"                       \
            "ands        r4, %2, #7       @ \n\t"                       \
            "beq         5f               @ \n\t"                       \
            "4:                           @ \n\t"                       \
            "subs        r4, r4, #1       @ \n\t"                       \
            "vld1.32     {d16[0]}, [r1]!  @ \n\t"                       \
            "vld1.32     {d0[0]}, [r0]!   @ \n\t"                       \
            "vmul.f32    q12, q8, q0      @ \n\t"                       \
            "vst1.32     {d24[0]}, [r2]!  @ \n\t"                       \
            "bne         4b               @ \n\t"                       \
            "5:                           @ "                           \
            :                                                           \
            :"r" (y),"r" (x),"r"(n)                                     \
            : "cc", "r0", "r1", "r2", "r4", "memory",                   \
              "q0", "q1", "q8", "q9", "q12", "q13",                     \
              "d0", "d1", "d2", "d3",                                   \
              "d16", "d17", "d18", "d19", "d24", "d25", "d26", "d27"    \
            );                                                          \
    }
#define THFloatVector_add(y, x, c, n) {                                 \
        float ctemp = c;                                                \
        float * caddr = &ctemp;                                         \
        __asm__ __volatile__ (                                          \
            "mov         r0, %0           @ \n\t"                       \
            "mov         r1, %1           @ \n\t"                       \
            "mov         r2, r0           @ \n\t"                       \
            "ldr         r5, [%2]         @ \n\t"                       \
            "vdup.32     q14, r5          @ \n\t"                       \
            "lsrs        r5, %3, #4       @ \n\t"                       \
            "beq         3f               @ \n\t"                       \
            "vld1.32     {d16-d19}, [r1]! @ \n\t"                       \
            "vld1.32     {d0-d3}, [r0]!   @ \n\t"                       \
            "vld1.32     {d20-d23}, [r1]! @ \n\t"                       \
            "vld1.32     {d4-d7}, [r0]!   @ \n\t"                       \
            "1:                           @ \n\t"                       \
            "vmla.f32    q0, q8, q14      @ \n\t"                       \
            "vmla.f32    q1, q9, q14      @ \n\t"                       \
            "vmla.f32    q2, q10, q14     @ \n\t"                       \
            "vmla.f32    q3, q11, q14     @ \n\t"                       \
            "subs        r5, r5, #1       @ \n\t"                       \
            "beq         2f               @ \n\t"                       \
            "vld1.32     {d16-d19}, [r1]! @ \n\t"                       \
            "vld1.32     {d20-d23}, [r1]! @ \n\t"                       \
            "vst1.32     {d0-d3}, [r2]!   @ \n\t"                       \
            "vld1.32     {d0-d3}, [r0]!   @ \n\t"                       \
            "vst1.32     {d4-d7}, [r2]!   @ \n\t"                       \
            "vld1.32     {d4-d7}, [r0]!   @ \n\t"                       \
            "b           1b               @ \n\t"                       \
            "2:                           @ \n\t"                       \
            "vst1.32     {d0-d3}, [r2]!   @ \n\t"                       \
            "vst1.32     {d4-d7}, [r2]!   @ \n\t"                       \
            "3:                           @ \n\t"                       \
            "lsrs        r5, %3, #3       @ \n\t"                       \
            "ands        r5, #1           @ \n\t"                       \
            "beq         4f               @ \n\t"                       \
            "vld1.32     {d16-d19}, [r1]! @ \n\t"                       \
            "vld1.32     {d0-d3}, [r0]!   @ \n\t"                       \
            "vmla.f32    q0, q8, q14      @ \n\t"                       \
            "vmla.f32    q1, q9, q14      @ \n\t"                       \
            "vst1.32     {d0-d3}, [r2]!   @ \n\t"                       \
            "4:                           @ \n\t"                       \
            "ands        r5, %3, #7       @ \n\t"                       \
            "beq         6f               @ \n\t"                       \
            "5:                           @ \n\t"                       \
            "subs        r5, r5, #1       @ \n\t"                       \
            "vld1.32     {d16[0]}, [r1]!  @ \n\t"                       \
            "vld1.32     {d0[0]}, [r0]!   @ \n\t"                       \
            "vmla.f32    d0, d16, d28     @ \n\t"                       \
            "vst1.32     d0[0], [r2]!     @ \n\t"                       \
            "bne         5b               @ \n\t"                       \
            "6:                           @ "                           \
            :                                                           \
            :"r" (y),"r" (x), "r"(caddr),"r"(n)                         \
            : "cc", "r0", "r1", "r2", "r5", "memory",                   \
              "q0", "q1", "q2", "q3", "q14",                            \
              "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",           \
              "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d28", "d29" \
            );                                                          \
    }

static inline void THDoubleVector_fill(double *x, const double c, const long n) {
  long i = 0;

  for(; i < n-4; i += 4)
  {
    x[i] = c;
    x[i+1] = c;
    x[i+2] = c;
    x[i+3] = c;
  }

  for(; i < n; i++)
    x[i] = c;
}

static inline void THDoubleVector_add(double *y, const double *x, const double c, const long n)
{
  long i = 0;

  for(;i < n-4; i += 4)
  {
    y[i] += c * x[i];
    y[i+1] += c * x[i+1];
    y[i+2] += c * x[i+2];
    y[i+3] += c * x[i+3];
  }

  for(; i < n; i++)
    y[i] += c * x[i];
}

static inline void THDoubleVector_diff(double *z, const double *x, const double *y, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    z[i] = x[i] - y[i];
    z[i+1] = x[i+1] - y[i+1];
    z[i+2] = x[i+2] - y[i+2];
    z[i+3] = x[i+3] - y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] - y[i];
}

static inline void THDoubleVector_scale(double *y, const double c, const long n)
{
  long i = 0;

  for(; i < n-4; i +=4)
  {
    y[i] *= c;
    y[i+1] *= c;
    y[i+2] *= c;
    y[i+3] *= c;
  }

  for(; i < n; i++)
    y[i] *= c;
}

static inline void THDoubleVector_mul(double *y, const double *x, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    y[i] *= x[i];
    y[i+1] *= x[i+1];
    y[i+2] *= x[i+2];
    y[i+3] *= x[i+3];
  }

  for(; i < n; i++)
    y[i] *= x[i];
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
