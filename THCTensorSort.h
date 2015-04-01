#ifndef TH_CUDA_TENSOR_SORT_INC
#define TH_CUDA_TENSOR_SORT_INC

#include "THCTensor.h"

THC_API void THCudaTensor_sort(THCState* state,
                               THCudaTensor* sorted,
                               THCudaTensor* indices,
                               THCudaTensor* input,
                               int dim, int order);

#endif
