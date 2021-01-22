#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathMagma.h"
#else

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

// MAGMA (i.e. CUDA implementation of LAPACK functions)
TORCH_CUDA_CU_API void THCTensor_(gels)(
    THCState* state,
    THCTensor* rb_,
    THCTensor* ra_,
    THCTensor* b_,
    THCTensor* a_);
TORCH_CUDA_CU_API void THCTensor_(
    geqrf)(THCState* state, THCTensor* ra_, THCTensor* rtau_, THCTensor* a_);

#endif // defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

#endif
