#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THVector.h"
#else

#include <ATen/core/Generator.h>
#include <ATen/core/DistributionsHelper.h>

TH_API void THVector_(fill)(scalar_t *x, const scalar_t c, const ptrdiff_t n);

#if !defined(TH_REAL_IS_BOOL) /* non bool only part */

TH_API void THVector_(cadd)(scalar_t *z, const scalar_t *x, const scalar_t *y, const scalar_t c, const ptrdiff_t n);
TH_API void THVector_(adds)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
TH_API void THVector_(cmul)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n);
TH_API void THVector_(muls)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
TH_API void THVector_(cdiv)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n);
TH_API void THVector_(divs)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
TH_API void THVector_(neg)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(normal_fill)(scalar_t *data,
                                                                   const int64_t size,
                                                                   struct at::Generator *generator,
                                                                   const scalar_t mean,
                                                                   const scalar_t stddev);

#endif /* non bool only part */

#if defined(TH_REAL_IS_SHORT) || defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
TH_API void THVector_(abs)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
#endif

/* floating point only now */
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

TH_API void THVector_(log1p)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(log2)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(sigmoid)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(exp)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(erf)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(erfc)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(cos)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(acos)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(cosh)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(sin)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(asin)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(sinh)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(tan)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(atan)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(tanh)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(pow)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
TH_API void THVector_(sqrt)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(rsqrt)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(ceil)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(floor)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(round)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(abs)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(trunc)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(frac)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(cinv)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);

#endif /* floating point only part */

#endif
