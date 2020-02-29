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

void THVector_(copy_DEFAULT)(scalar_t *x, const scalar_t *y, const ptrdiff_t n) {
  ptrdiff_t i = 0;

  for(; i <n-4; i+=4)
  {
    x[i] = y[i];
    x[i+1] = y[i+1];
    x[i+2] = y[i+2];
    x[i+3] = y[i+3];
  }

  for(; i < n; i++)
    x[i] = y[i];
}

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

#define VECTOR_IMPLEMENT_FUNCTION(NAME, CFUNC)  \
  void THVector_(NAME)(scalar_t *y, const scalar_t *x, const ptrdiff_t n) \
  { \
    ptrdiff_t i = 0;  \
    for(; i<n-4; i+=4)  \
    { \
      y[i] = CFUNC(x[i]); \
      y[i+1] = CFUNC(x[i+1]); \
      y[i+2] = CFUNC(x[i+2]); \
      y[i+3] = CFUNC(x[i+3]); \
    } \
    for(; i < n; i++) \
      y[i] = CFUNC(x[i]); \
  } \

#define VECTOR_IMPLEMENT_FUNCTION_VALUE(NAME, CFUNC)  \
  void THVector_(NAME)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n) \
  { \
    ptrdiff_t i = 0;  \
    for(; i<n-4; i+=4)  \
    { \
      y[i] = CFUNC(x[i], c);  \
      y[i+1] = CFUNC(x[i+1], c);  \
      y[i+2] = CFUNC(x[i+2], c);  \
      y[i+3] = CFUNC(x[i+3], c);  \
    } \
    for(; i < n; i++) \
      y[i] = CFUNC(x[i], c);  \
  } \

/* floating point only now */
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

#if defined (TH_REAL_IS_FLOAT)
#define TH_MATH_NAME(fn) fn##f
#else
#define TH_MATH_NAME(fn) fn
#endif

VECTOR_IMPLEMENT_FUNCTION(erf,TH_MATH_NAME(erf))
VECTOR_IMPLEMENT_FUNCTION(erfc,TH_MATH_NAME(erfc))
VECTOR_IMPLEMENT_FUNCTION(cos,TH_MATH_NAME(cos))
VECTOR_IMPLEMENT_FUNCTION(cosh,TH_MATH_NAME(cosh))
VECTOR_IMPLEMENT_FUNCTION(tan,TH_MATH_NAME(tan))
VECTOR_IMPLEMENT_FUNCTION(atan,TH_MATH_NAME(atan))
VECTOR_IMPLEMENT_FUNCTION(tanh,TH_MATH_NAME(tanh))
VECTOR_IMPLEMENT_FUNCTION_VALUE(pow,TH_MATH_NAME(pow))

#undef TH_MATH_NAME
#endif /* floating point only part */

VECTOR_IMPLEMENT_FUNCTION(neg,-)

#endif /* non bool only part */

#endif
