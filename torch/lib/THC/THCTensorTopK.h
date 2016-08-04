#ifndef TH_CUDA_TENSOR_TOPK_INC
#define TH_CUDA_TENSOR_TOPK_INC

#include "THCTensor.h"

/* Returns the set of all kth smallest (or largest) elements, depending */
/* on `dir` */
THC_API void THCudaTensor_topk(THCState* state,
                               THCudaTensor* topK,
                               THCudaTensor* indices,
                               THCudaTensor* input,
                               long k, int dim, int dir, int sorted);

#endif
