#ifndef TH_HALF_H
#define TH_HALF_H

#include "THGeneral.h"
#include <stdint.h>

/* Neither built-in nor included from Cutorch, use our definition lifted from CUDA */
#if defined(__GNUC__)
#define __thalign__(n) __attribute__((aligned(n)))
#elif defined(_WIN32)
#define __thalign__(n) __declspec(align(n))
#else
#define __thalign__(n)
#endif

typedef struct __thalign__(2){
  unsigned short x;
} __THHalf;

typedef struct __thalign__(4) {
  unsigned int x;
} __THHalf2;

typedef __THHalf THHalf;
typedef __THHalf2 THHalf2;

TH_API void TH_float2halfbits(float*, unsigned short*);
TH_API void TH_halfbits2float(unsigned short*, float*);

TH_API THHalf TH_float2half(float);
TH_API float  TH_half2float(THHalf);

#ifndef TH_HALF_BITS_TO_LITERAL
# define TH_HALF_BITS_TO_LITERAL(n) { n }
#endif

#define TH_HALF_ZERO 0x0U
#define TH_HALF_INF  0x7C00U

#undef __thalign__
#endif
