#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THVector.h"
#else

struct THGenerator;

TH_API void THVector_(fill)(real *x, const real c, const ptrdiff_t n);
TH_API void THVector_(cadd)(real *z, const real *x, const real *y, const real c, const ptrdiff_t n);
TH_API void THVector_(adds)(real *y, const real *x, const real c, const ptrdiff_t n);
TH_API void THVector_(cmul)(real *z, const real *x, const real *y, const ptrdiff_t n);
TH_API void THVector_(muls)(real *y, const real *x, const real c, const ptrdiff_t n);
TH_API void THVector_(cdiv)(real *z, const real *x, const real *y, const ptrdiff_t n);
TH_API void THVector_(divs)(real *y, const real *x, const real c, const ptrdiff_t n);
TH_API void THVector_(copy)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(neg)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(normal_fill)(real *data,
                                   const int64_t size,
                                   struct THGenerator *generator,
                                   const real mean,
                                   const real stddev);

#if defined(TH_REAL_IS_SHORT) || defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
TH_API void THVector_(abs)(real *y, const real *x, const ptrdiff_t n);
#endif

/* floating point only now */
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

TH_API void THVector_(log)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(lgamma)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(digamma)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(trigamma)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(log10)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(log1p)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(log2)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(sigmoid)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(exp)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(expm1)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(erf)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(erfinv)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(cos)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(acos)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(cosh)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(sin)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(asin)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(sinh)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(tan)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(atan)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(tanh)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(pow)(real *y, const real *x, const real c, const ptrdiff_t n);
TH_API void THVector_(sqrt)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(rsqrt)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(ceil)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(floor)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(round)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(abs)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(trunc)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(frac)(real *y, const real *x, const ptrdiff_t n);
TH_API void THVector_(cinv)(real *y, const real *x, const ptrdiff_t n);

#endif /* floating point only part */

#endif
