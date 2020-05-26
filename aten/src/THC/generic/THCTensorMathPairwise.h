#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathPairwise.h"
#else

#if !defined(THC_REAL_IS_BOOL)

THC_API void THCTensor_(mul)(THCState *state, THCTensor *self, THCTensor *src, scalar_t value);
THC_API void THCTensor_(fmod)(THCState *state, THCTensor *self, THCTensor *src, scalar_t value);

#endif

#endif
