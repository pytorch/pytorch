#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZVectorDefault.c"
#else

#include "THZTypeMacros.h"

void THZVector_(copy_DEFAULT)(ntype *x, const ntype *y, const ptrdiff_t n) {
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

void THZVector_(fill_DEFAULT)(ntype *x, const ntype c, const ptrdiff_t n) {
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

void THZVector_(cadd_DEFAULT)(ntype *z, const ntype *x, const ntype *y, const ntype c, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i<n-4; i+=4)
  {
    z[i] = x[i] + c * y[i];
    z[i+1] = x[i+1] + c * y[i+1];
    z[i+2] = x[i+2] + c * y[i+2];
    z[i+3] = x[i+3] + c * y[i+3];
  }

  for(; i<n; i++)
    z[i] = x[i] + c * y[i];
}

void THZVector_(adds_DEFAULT)(ntype *y, const ntype *x, const ntype c, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i<n-4; i+=4)
  {
    y[i] = x[i] + c;
    y[i+1] = x[i+1] + c;
    y[i+2] = x[i+2] + c;
    y[i+3] = x[i+3] + c;
  }

  for(; i<n; i++)
    y[i] = x[i] + c;
}

void THZVector_(cmul_DEFAULT)(ntype *z, const ntype *x, const ntype *y, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i <n-4; i+=4)
  {
    z[i] = x[i] * y[i];
    z[i+1] = x[i+1] * y[i+1];
    z[i+2] = x[i+2] * y[i+2];
    z[i+3] = x[i+3] * y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] * y[i];
}

void THZVector_(muls_DEFAULT)(ntype *y, const ntype *x, const ntype c, const ptrdiff_t n)
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

void THZVector_(cdiv_DEFAULT)(ntype *z, const ntype *x, const ntype *y, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i<n-4; i+=4)
  {
    z[i] = x[i] / y[i];
    z[i+1] = x[i+1] / y[i+1];
    z[i+2] = x[i+2] / y[i+2];
    z[i+3] = x[i+3] / y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] / y[i];
}

void THZVector_(divs_DEFAULT)(ntype *y, const ntype *x, const ntype c, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i<n-4; i+=4)
  {
    y[i] = x[i] / c;
    y[i+1] = x[i+1] / c;
    y[i+2] = x[i+2] / c;
    y[i+3] = x[i+3] / c;
  }

  for(; i < n; i++)
    y[i] = x[i] / c;
}

#define VECTOR_IMPLEMENT_FUNCTION(NAME, CFUNC)  \
  void THZVector_(NAME)(ntype *y, const ntype *x, const ptrdiff_t n) \
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

#define VECTOR_IMPLEMENT_FUNCTION_PART(NAME, CFUNC)  \
  void THZVector_(NAME)(part *y, const ntype *x, const ptrdiff_t n) \
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
  void THZVector_(NAME)(ntype *y, const ntype *x, const ntype c, const ptrdiff_t n) \
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


VECTOR_IMPLEMENT_FUNCTION(log,THZ_MATH_NAME(log))
VECTOR_IMPLEMENT_FUNCTION(exp,THZ_MATH_NAME(exp))
VECTOR_IMPLEMENT_FUNCTION(cos,THZ_MATH_NAME(cos))
VECTOR_IMPLEMENT_FUNCTION(acos,THZ_MATH_NAME(acos))
VECTOR_IMPLEMENT_FUNCTION(cosh,THZ_MATH_NAME(cosh))
VECTOR_IMPLEMENT_FUNCTION(sin,THZ_MATH_NAME(sin))
VECTOR_IMPLEMENT_FUNCTION(asin,THZ_MATH_NAME(asin))
VECTOR_IMPLEMENT_FUNCTION(sinh,THZ_MATH_NAME(sinh))
VECTOR_IMPLEMENT_FUNCTION(tan,THZ_MATH_NAME(tan))
VECTOR_IMPLEMENT_FUNCTION(atan,THZ_MATH_NAME(atan))
VECTOR_IMPLEMENT_FUNCTION(tanh,THZ_MATH_NAME(tanh))
VECTOR_IMPLEMENT_FUNCTION_VALUE(pow,THZ_MATH_NAME(pow))
VECTOR_IMPLEMENT_FUNCTION(sqrt,THZ_MATH_NAME(sqrt))

// additional math functions
VECTOR_IMPLEMENT_FUNCTION(sigmoid,THZMath_(sigmoid))
VECTOR_IMPLEMENT_FUNCTION(rsqrt,THZMath_(rsqrt))

VECTOR_IMPLEMENT_FUNCTION_PART(abs,THZ_MATH_NAME(abs))
VECTOR_IMPLEMENT_FUNCTION_PART(real,THZ_MATH_NAME(real))
VECTOR_IMPLEMENT_FUNCTION_PART(imag,THZ_MATH_NAME(imag))
VECTOR_IMPLEMENT_FUNCTION_PART(arg, THZ_MATH_NAME(arg))
VECTOR_IMPLEMENT_FUNCTION_PART(proj,THZ_MATH_NAME(proj))
// c conjugate is the only function
// that does not use c as name prefix
VECTOR_IMPLEMENT_FUNCTION(conj, THZ_MATH_NAME(onj))
VECTOR_IMPLEMENT_FUNCTION(cinv, 1.0 / )

// addtional math functions
VECTOR_IMPLEMENT_FUNCTION(log1p,THZMath_(log1p))

VECTOR_IMPLEMENT_FUNCTION(neg,-)
#endif
