#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMath.h"
#else

THC_API void THCTensor_(fill)(THCState *state, THCTensor *self, scalar_t value);
THC_API void THCTensor_(zero)(THCState *state, THCTensor *self);
THC_API ptrdiff_t THCTensor_(numel)(THCState *state, THCTensor *t);


#endif
