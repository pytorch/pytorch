#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THVectorDefault.cpp"
#else

#include <ATen/Context.h>

void THVector_(fill_DEFAULT)(scalar_t *x, const scalar_t c, const ptrdiff_t n) {
  ptrdiff_t i = 0;

  for(; i <n-4; i+=4)
  {
    x[i] = c;
    x[i+1] = c;
    x[i+2] = c;
    x[i+3] = c;
  }

  for(; i < n; i++)
    x[i] = c;
}

#if !defined(TH_REAL_IS_BOOL) /* non bool only part */

void THVector_(muls_DEFAULT)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i <n-4; i+=4)
  {
    y[i] = x[i] * c;
    y[i+1] = x[i+1] * c;
    y[i+2] = x[i+2] * c;
    y[i+3] = x[i+3] * c;
  }

  for(; i < n; i++)
    y[i] = x[i] * c;
}

#endif /* non bool only part */

#endif
