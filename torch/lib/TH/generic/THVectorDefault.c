#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THVectorDefault.c"
#else

void THVector_(copy_DEFAULT)(real *x, const real *y, const ptrdiff_t n) {
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

void THVector_(fill_DEFAULT)(real *x, const real c, const ptrdiff_t n) {
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

void THVector_(cadd_DEFAULT)(real *z, const real *x, const real *y, const real c, const ptrdiff_t n)
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

void THVector_(adds_DEFAULT)(real *y, const real *x, const real c, const ptrdiff_t n)
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

void THVector_(cmul_DEFAULT)(real *z, const real *x, const real *y, const ptrdiff_t n)
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

void THVector_(muls_DEFAULT)(real *y, const real *x, const real c, const ptrdiff_t n)
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

void THVector_(cdiv_DEFAULT)(real *z, const real *x, const real *y, const ptrdiff_t n)
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

void THVector_(divs_DEFAULT)(real *y, const real *x, const real c, const ptrdiff_t n)
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
  void THVector_(NAME)(real *y, const real *x, const ptrdiff_t n) \
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
  void THVector_(NAME)(real *y, const real *x, const real c, const ptrdiff_t n) \
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

#if defined(TH_REAL_IS_LONG)
VECTOR_IMPLEMENT_FUNCTION(abs,labs)
#endif /* long only part */

#if defined(TH_REAL_IS_SHORT) || defined(TH_REAL_IS_INT)
VECTOR_IMPLEMENT_FUNCTION(abs,abs)
#endif /* int only part */


/* floating point only now */
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

#if defined (TH_REAL_IS_FLOAT)
#define TH_MATH_NAME(fn) fn##f
#else
#define TH_MATH_NAME(fn) fn
#endif

VECTOR_IMPLEMENT_FUNCTION(log,TH_MATH_NAME(log))
VECTOR_IMPLEMENT_FUNCTION(lgamma,TH_MATH_NAME(lgamma))
VECTOR_IMPLEMENT_FUNCTION(log1p,TH_MATH_NAME(log1p))
VECTOR_IMPLEMENT_FUNCTION(sigmoid,TH_MATH_NAME(TH_sigmoid))
VECTOR_IMPLEMENT_FUNCTION(exp,TH_MATH_NAME(exp))
VECTOR_IMPLEMENT_FUNCTION(cos,TH_MATH_NAME(cos))
VECTOR_IMPLEMENT_FUNCTION(acos,TH_MATH_NAME(acos))
VECTOR_IMPLEMENT_FUNCTION(cosh,TH_MATH_NAME(cosh))
VECTOR_IMPLEMENT_FUNCTION(sin,TH_MATH_NAME(sin))
VECTOR_IMPLEMENT_FUNCTION(asin,TH_MATH_NAME(asin))
VECTOR_IMPLEMENT_FUNCTION(sinh,TH_MATH_NAME(sinh))
VECTOR_IMPLEMENT_FUNCTION(tan,TH_MATH_NAME(tan))
VECTOR_IMPLEMENT_FUNCTION(atan,TH_MATH_NAME(atan))
VECTOR_IMPLEMENT_FUNCTION(tanh,TH_MATH_NAME(tanh))
VECTOR_IMPLEMENT_FUNCTION_VALUE(pow,TH_MATH_NAME(pow))
VECTOR_IMPLEMENT_FUNCTION(sqrt,TH_MATH_NAME(sqrt))
VECTOR_IMPLEMENT_FUNCTION(rsqrt,TH_MATH_NAME(TH_rsqrt))
VECTOR_IMPLEMENT_FUNCTION(ceil,TH_MATH_NAME(ceil))
VECTOR_IMPLEMENT_FUNCTION(floor,TH_MATH_NAME(floor))
VECTOR_IMPLEMENT_FUNCTION(round,TH_MATH_NAME(round))
VECTOR_IMPLEMENT_FUNCTION(abs,TH_MATH_NAME(fabs))
VECTOR_IMPLEMENT_FUNCTION(trunc,TH_MATH_NAME(trunc))
VECTOR_IMPLEMENT_FUNCTION(frac,TH_MATH_NAME(TH_frac))
VECTOR_IMPLEMENT_FUNCTION(cinv, TH_MATH_NAME(1.0) / )

#undef TH_MATH_NAME
#endif /* floating point only part */

#ifndef TH_REAL_IS_BYTE
VECTOR_IMPLEMENT_FUNCTION(neg,-)
#endif

#endif
