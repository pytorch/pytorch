#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMath.h"
#else

THC_API void THCTensor_(fill)(THCState *state, THCTensor *self, real value);
THC_API void THCTensor_(zero)(THCState *state, THCTensor *self);

THC_API void THCTensor_(zeros)(THCState *state, THCTensor *r_, THLongStorage *size);
THC_API void THCTensor_(ones)(THCState *state, THCTensor *r_, THLongStorage *size);
THC_API void THCTensor_(reshape)(THCState *state, THCTensor *r_, THCTensor *t, THLongStorage *size);
THC_API ptrdiff_t THCTensor_(numel)(THCState *state, THCTensor *t);
THC_API void THCTensor_(cat)(THCState *state, THCTensor *result, THCTensor *ta, THCTensor *tb, int dimension);
THC_API void THCTensor_(catArray)(THCState *state, THCTensor *result, THCTensor **inputs, int numInputs, int dimension);
THC_API void THCTensor_(nonzero)(THCState* state, THCudaLongTensor *tensor, THCTensor *self);


#endif
