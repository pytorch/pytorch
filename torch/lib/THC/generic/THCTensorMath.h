#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMath.h"
#else

THC_API void THCTensor_(fill)(THCState *state, THCTensor *self, real value);
THC_API void THCTensor_(zero)(THCState *state, THCTensor *self);

THC_API void THCTensor_(zeros)(THCState *state, THCTensor *r_, THLongStorage *size);
THC_API void THCTensor_(zerosLike)(THCState *state, THCTensor *r_, THCTensor* input);
THC_API void THCTensor_(ones)(THCState *state, THCTensor *r_, THLongStorage *size);
THC_API void THCTensor_(onesLike)(THCState *state, THCTensor *r_, THCTensor* input);
THC_API void THCTensor_(reshape)(THCState *state, THCTensor *r_, THCTensor *t, THLongStorage *size);
THC_API ptrdiff_t THCTensor_(numel)(THCState *state, THCTensor *t);
THC_API void THCTensor_(cat)(THCState *state, THCTensor *result, THCTensor *ta, THCTensor *tb, int dimension);
THC_API void THCTensor_(catArray)(THCState *state, THCTensor *result, THCTensor **inputs, int numInputs, int dimension);
THC_API void THCTensor_(nonzero)(THCState* state, THCudaLongTensor *tensor, THCTensor *self);

THC_API void THCTensor_(tril)(THCState *state, THCTensor *self, THCTensor *src, int64_t k);
THC_API void THCTensor_(triu)(THCState *state, THCTensor *self, THCTensor *src, int64_t k);
THC_API void THCTensor_(diag)(THCState *state, THCTensor *self, THCTensor *src, int64_t k);
THC_API void THCTensor_(eye)(THCState *state, THCTensor *self, int64_t n, int64_t k);

THC_API accreal THCTensor_(trace)(THCState *state, THCTensor *self);

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

THC_API void THCTensor_(linspace)(THCState *state, THCTensor *r_, real a, real b, int64_t n);
THC_API void THCTensor_(logspace)(THCState *state, THCTensor *r_, real a, real b, int64_t n);

#endif

THC_API void THCTensor_(range)(THCState *state, THCTensor *r_, accreal xmin, accreal xmax, accreal step);
THC_API void THCTensor_(arange)(THCState *state, THCTensor *r_, accreal xmin, accreal xmax, accreal step);

#endif
