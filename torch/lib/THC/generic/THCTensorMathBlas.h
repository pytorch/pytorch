#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathBlas.h"
#else

THC_API real THCTensor_(dot)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(addmv)(THCState *state, THCTensor *self, real beta, THCTensor *t, real alpha, THCTensor *mat, THCTensor *vec);
THC_API void THCTensor_(addmm)(THCState *state, THCTensor *self, real beta, THCTensor *t, real alpha, THCTensor *mat1, THCTensor *mat2);
THC_API void THCTensor_(addr)(THCState *state, THCTensor *self, real beta, THCTensor *t, real alpha, THCTensor *vec1, THCTensor *vec2);
THC_API void THCTensor_(addbmm)(THCState *state, THCTensor *result, real beta, THCTensor *t, real alpha, THCTensor *batch1, THCTensor *batch2);
THC_API void THCTensor_(baddbmm)(THCState *state, THCTensor *result, real beta, THCTensor *t, real alpha, THCTensor *batch1, THCTensor *batch2);


#endif
