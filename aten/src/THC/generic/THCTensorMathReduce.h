#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathReduce.h"
#else

#if !defined(THC_REAL_IS_BOOL)

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

TORCH_CUDA_CU_API void THCTensor_(renorm)(
    THCState* state,
    THCTensor* self,
    THCTensor* src,
    scalar_t value,
    int dimension,
    scalar_t max_norm);
TORCH_CUDA_CU_API void THCTensor_(norm)(
    THCState* state,
    THCTensor* self,
    THCTensor* src,
    scalar_t value,
    int dimension,
    int keepdim);

TORCH_CUDA_CU_API accreal
    THCTensor_(std_all)(THCState* state, THCTensor* self, bool unbiased);
TORCH_CUDA_CU_API accreal
    THCTensor_(normall)(THCState* state, THCTensor* self, scalar_t value);
TORCH_CUDA_CU_API accreal
    THCTensor_(var_all)(THCState* state, THCTensor* self, bool unbiased);

#endif

TORCH_CUDA_CU_API void THCTensor_(prod)(
    THCState* state,
    THCTensor* self,
    THCTensor* src,
    int dim,
    int keepdim);

#endif

#endif
