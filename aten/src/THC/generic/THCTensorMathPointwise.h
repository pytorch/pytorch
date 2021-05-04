#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathPointwise.h"
#else

#if !defined(THC_REAL_IS_BOOL)

TORCH_CUDA_CU_API void THCTensor_(crossKernel)(
    THCState* state,
    THCTensor* self,
    THCTensor* src1,
    THCTensor* src2,
    int dimension);

#endif
#endif
