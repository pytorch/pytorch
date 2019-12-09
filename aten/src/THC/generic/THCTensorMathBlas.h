#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathBlas.h"
#else

THC_API accreal THCTensor_(dot)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(addmv)(THCState *state, THCTensor *self, THCTensor *t, THCTensor *mat, THCTensor *vec, scalar_t beta, scalar_t alpha);
THC_API void THCTensor_(addmm)(THCState *state, THCTensor *self, THCTensor *t, THCTensor *mat1, THCTensor *mat2, scalar_t beta, scalar_t alpha);
THC_API void THCTensor_(addr)(THCState *state, THCTensor *self, THCTensor *t, THCTensor *vec1, THCTensor *vec2, scalar_t beta, scalar_t alpha);
THC_API void THCTensor_(addbmm)(THCState *state, THCTensor *result, THCTensor *t, THCTensor *batch1, THCTensor *batch2, scalar_t beta, scalar_t alpha);
THC_API void THCTensor_(baddbmm)(THCState *state, THCTensor *result, THCTensor *t, THCTensor *batch1, THCTensor *batch2, scalar_t beta, scalar_t alpha);

#endif
