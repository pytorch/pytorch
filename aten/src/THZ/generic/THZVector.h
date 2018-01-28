#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZVector.h"
#else

TH_API void THZVector_(fill)(ntype *x, const ntype c, const ptrdiff_t n);
TH_API void THZVector_(cadd)(ntype *z, const ntype *x, const ntype *y, const ntype c, const ptrdiff_t n);
TH_API void THZVector_(adds)(ntype *y, const ntype *x, const ntype c, const ptrdiff_t n);
TH_API void THZVector_(cmul)(ntype *z, const ntype *x, const ntype *y, const ptrdiff_t n);
TH_API void THZVector_(muls)(ntype *y, const ntype *x, const ntype c, const ptrdiff_t n);
TH_API void THZVector_(cdiv)(ntype *z, const ntype *x, const ntype *y, const ptrdiff_t n);
TH_API void THZVector_(divs)(ntype *y, const ntype *x, const ntype c, const ptrdiff_t n);
TH_API void THZVector_(copy)(ntype *y, const ntype *x, const ptrdiff_t n);

/* floating point only now */

TH_API void THZVector_(log)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(log1p)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(sigmoid)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(exp)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(cos)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(acos)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(cosh)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(sin)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(asin)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(sinh)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(tan)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(atan)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(tanh)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(pow)(ntype *y, const ntype *x, const ntype c, const ptrdiff_t n);
TH_API void THZVector_(sqrt)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(rsqrt)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(cinv)(ntype *y, const ntype *x, const ptrdiff_t n);

TH_API void THZVector_(abs)(part *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(real)(part *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(imag)(part *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(arg)(part *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(proj)(part *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(conj)(ntype *y, const ntype *x, const ptrdiff_t n);
TH_API void THZVector_(neg)(ntype *y, const ntype *x, const ptrdiff_t n);

/* Initialize the dispatch pointers */
TH_API void THZVector_(vectorDispatchInit)(void);

#endif
