#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THVector.h"
#else

#include <ATen/core/DistributionsHelper.h>

TH_API void THVector_(fill)(scalar_t *x, const scalar_t c, const ptrdiff_t n);

#if !defined(TH_REAL_IS_BOOL) /* non bool only part */

TH_API void THVector_(muls)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);
TH_API void THVector_(neg)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);

#endif /* non bool only part */

/* floating point only now */
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

TH_API void THVector_(exp)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(erf)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(erfc)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(cos)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(cosh)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(tan)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(atan)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(tanh)(scalar_t *y, const scalar_t *x, const ptrdiff_t n);
TH_API void THVector_(pow)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n);

#endif /* floating point only part */

#endif
