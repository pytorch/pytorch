#ifndef TH_CUDA_TENSOR_SORT_INC
#define TH_CUDA_TENSOR_SORT_INC

#include "THCTensor.h"

/* Performs an in-place sort of (keys, values). Only works for slice sizes
   <= 2048 at the moment (slice size == size of keys/values dim `dim`) */
THC_API void THCudaTensor_sortKeyValueInplace(THCState* state,
                                              THCudaTensor* keys,
                                              THCudaTensor* values,
                                              int dim, int order);

/* Performs an out-of-place sort of `input`, returning the per-slice indices
   in `indices` and the sorted values in `sorted` */
THC_API void THCudaTensor_sort(THCState* state,
                               THCudaTensor* sorted,
                               THCudaTensor* indices,
                               THCudaTensor* input,
                               int dim, int order);

#endif
