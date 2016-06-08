#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathPointwise.h"
#else

THC_API void THCTensor_(cadd)(THCState *state, THCTensor *self, THCTensor *src1, real value, THCTensor *src2);
THC_API void THCTensor_(csub)(THCState *state, THCTensor *self, THCTensor *src1, real value, THCTensor *src2);
THC_API void THCTensor_(cmul)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(cpow)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(cdiv)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2);

#endif
