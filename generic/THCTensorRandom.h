#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorRandom.h"
#else

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

THC_API void THCTensor_(uniform)(struct THCState *state, THCTensor *self, double a, double b);
THC_API void THCTensor_(rand)(THCState *state, THCTensor *r_, THLongStorage *size);
THC_API void THCTensor_(normal)(struct THCState *state, THCTensor *self, double mean, double stdv);

#endif

THC_API void THCTensor_(bernoulli)(struct THCState *state, THCTensor *self, double p);

#endif
