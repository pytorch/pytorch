#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathBlas.h"
#else

THC_API void THCTensor_(baddbmm)(THCState *state, THCTensor *result, THCTensor *t, THCTensor *batch1, THCTensor *batch2, scalar_t beta, scalar_t alpha);

#endif
