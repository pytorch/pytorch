#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THVector.h"
#else

TH_API void THVector_(fill)(real *x, const real c, const ptrdiff_t n);
TH_API void THVector_(add)(real *y, const real *x, const real c, const ptrdiff_t n);
TH_API void THVector_(diff)(real *z, const real *x, const real *y, const ptrdiff_t n);
TH_API void THVector_(scale)(real *y, const real c, const ptrdiff_t n);
TH_API void THVector_(mul)(real *y, const real *x, const ptrdiff_t n);

/* Initialize the dispatch pointers */
TH_API void THVector_(vectorDispatchInit)();

#endif
