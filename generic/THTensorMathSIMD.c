#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMathSIMD.c"
#else

#ifndef _MSC_VER
#include <x86intrin.h>
#else
#include <intrin.h>
#endif

#if defined(TH_REAL_IS_DOUBLE)

void THTensor_(add_AVX)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
    real *rp = THTensor_(data)(r_);
    real *tp = THTensor_(data)(t);
    ptrdiff_t sz = THTensor_(nElement)(t);
    ptrdiff_t i = 0;
    __m256d YMM3 = _mm256_set_pd(value, value, value, value);
    __m256d YMM0, YMM2;
    for (; i<=((sz)-4); i+=4) {
      YMM0 = _mm256_loadu_pd(tp+i);
      YMM2 = _mm256_add_pd(YMM0, YMM3);
      _mm256_storeu_pd(rp+i, YMM2);
    }
    for (; i<sz; i++) {
      rp[i] = tp[i] + value;
    }
  } else {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data + value;);
  }
}

#endif

#endif
