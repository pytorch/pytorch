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

// dense = sparse * dense
TH_API void THSTensor_(spmm)(THTensor *r_, THSTensor *sparse, THTensor *dense);
// sparse = sparse * dense
TH_API void THSTensor_(sspmm)(THSTensor *r_, THSTensor *sparse, THTensor *dense);
TH_API void THSTensor_(spcadd)(THTensor *r_, THTensor *dense, real value, THSTensor *sparse);

#endif

