#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathPairwise.h"
#else

THC_API void THCTensor_(add)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(sub)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(mul)(THCState *state, THCTensor *self, THCTensor *src, real value);
THC_API void THCTensor_(div)(THCState *state, THCTensor *self, THCTensor *src, real value);

#endif
