#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "generic/THSTensorMath.h"
#else

/* The convention is:
 * spOP has one of the OP as a sparse tensor
 * sspOP returns a sparse result
 * spOPs has all arguments sparse
 *
 * Everything is up to discretion
 */

TH_API void THSTensor_(zero)(THSTensor *r_);
TH_API void THSTensor_(zeros)(THSTensor *r_, THLongStorage *size);
TH_API void THSTensor_(zerosLike)(THSTensor *r_, THSTensor *input);

TH_API void THSTensor_(mul)(THSTensor *r_, THSTensor *t, real value);
TH_API void THSTensor_(div)(THSTensor *r_, THSTensor *t, real value);
TH_API void THSTensor_(cadd)(THSTensor *r_, THSTensor *t, real value, THSTensor *src);
TH_API void THSTensor_(csub)(THSTensor *r_, THSTensor *t, real value, THSTensor *src);
TH_API void THSTensor_(cmul)(THSTensor *r_, THSTensor *t, THSTensor *src);

TH_API void THTensor_(spaddcmul)(THTensor *r_, THTensor *t, real value, THSTensor *src1, THSTensor *src2);

// dense = beta * dense + alpha * sparse * dense
TH_API void THSTensor_(spaddmm)(THTensor *r_, real beta, THTensor *t, real alpha, THSTensor *sparse, THTensor *dense);
// sparse = beta * sparse + alpha * sparse * dense
TH_API void THSTensor_(sspaddmm)(THSTensor *r_, real beta, THSTensor *t, real alpha, THSTensor *sparse, THTensor *dense);
// hybrid = alpha * sparse * dense
TH_API void THSTensor_(hspmm)(THSTensor *r_, real alpha, THSTensor *sparse, THTensor *dense);
TH_API void THSTensor_(spcadd)(THTensor *r_, THTensor *dense, real value, THSTensor *sparse);

TH_API void THSTensor_(pow)(THSTensor *r_, THSTensor *t, real value);

#if defined(THS_REAL_IS_FLOAT) || defined(THS_REAL_IS_DOUBLE)
TH_API accreal THSTensor_(normall)(THSTensor *self, real value);
#endif

#endif
