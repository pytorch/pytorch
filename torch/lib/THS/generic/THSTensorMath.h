#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "generic/THSTensorMath.h"
#else

/* The convention is:
 * spOP has one of the OP as a sparse tensor
 * sspOP returns a sparse result
 * spOPs has all arguments sparse
 *
 * Everything is is up to discretion
 */

// dense = beta * dense + alpha * sparse * dense
TH_API void THSTensor_(spaddmm)(THTensor *r_, real beta, THTensor *t, real alpha, THSTensor *sparse, THTensor *dense);
// sparse = beta * sparse + alpha * sparse * dense
TH_API void THSTensor_(sspaddmm)(THSTensor *r_, real beta, THSTensor *t, real alpha, THSTensor *sparse, THTensor *dense);
TH_API void THSTensor_(spcadd)(THTensor *r_, THTensor *dense, real value, THSTensor *sparse);

#endif

