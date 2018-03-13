#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensorMath.h"
#else

/* The convention is:
 * spOP has one of the OP as a sparse tensor
 * sspOP returns a sparse result
 * spOPs has all arguments sparse
 *
 * Everything is up to discretion
 */

TH_API void THCSTensor_(zero)(THCState *state, THCSTensor *r_);
TH_API void THCSTensor_(zeros)(THCState *state, THCSTensor *r_, THLongStorage *size);
TH_API void THCSTensor_(zerosLike)(THCState *state, THCSTensor *r_, THCSTensor *input);

TH_API void THCTensor_(spaddcmul)(THCState *state, THCTensor *r_, THCTensor *t, real value, THCSTensor *src1, THCSTensor *src2);
TH_API void THCTensor_(spaddcdiv)(THCState *state, THCTensor *r_, THCTensor *t, real value, THCSTensor *src1, THCSTensor *src2);

// dense = beta * dense + alpha * sparse * dense
TH_API void THCSTensor_(spaddmm)(THCState *state, THCTensor *r_, real beta, THCTensor *t, real alpha, THCSTensor *sparse, THCTensor *dense);
// sparse = beta * sparse + alpha * sparse * dense
TH_API void THCSTensor_(sspaddmm)(THCState *state, THCSTensor *r_, real beta, THCSTensor *t, real alpha, THCSTensor *sparse, THCTensor *dense);
// hybrid = alpha * sparse * dense
TH_API void THCSTensor_(hspmm)(THCState *state, THCSTensor *r_, real alpha, THCSTensor *sparse, THCTensor *dense);
TH_API void THCSTensor_(spcadd)(THCState *state, THCTensor *r_, THCTensor *dense, real value, THCSTensor *sparse);
TH_API void THCSTensor_(mul)(THCState *state, THCSTensor *r_, THCSTensor *t, real value);
TH_API void THCSTensor_(div)(THCState *state, THCSTensor *r_, THCSTensor *t, real value);
TH_API void THCSTensor_(cadd)(THCState *state, THCSTensor *r_, THCSTensor *t, real value, THCSTensor *src);
TH_API void THCSTensor_(csub)(THCState *state, THCSTensor *r_, THCSTensor *t, real value, THCSTensor *src);
TH_API void THCSTensor_(cmul)(THCState *state, THCSTensor *r_, THCSTensor *t, THCSTensor *src);
TH_API void THCSTensor_(pow)(THCState *state, THCSTensor *r_, THCSTensor *t, real value);

#if defined(THCS_REAL_IS_FLOAT) || defined(THCS_REAL_IS_DOUBLE) || defined(THCS_REAL_IS_HALF)
TH_API accreal THCSTensor_(normall)(THCState *state, THCSTensor *self, real value);
#endif

#endif
