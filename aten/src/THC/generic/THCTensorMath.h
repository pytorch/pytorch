#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMath.h"
#else

THC_API void THCTensor_(fill)(THCState *state, THCTensor *self, scalar_t value);
THC_API void THCTensor_(zero)(THCState *state, THCTensor *self);
THC_API void THCTensor_(nonzero)(THCState* state, THCudaLongTensor *tensor, THCTensor *self);
THC_API ptrdiff_t THCTensor_(numel)(THCState *state, THCTensor *t);

#if !defined(THC_REAL_IS_BOOL) /* non bool only part */

THC_API void THCTensor_(diag)(THCState *state, THCTensor *self, THCTensor *src, int64_t k);
THC_API accreal THCTensor_(trace)(THCState *state, THCTensor *self);

#endif

#endif
