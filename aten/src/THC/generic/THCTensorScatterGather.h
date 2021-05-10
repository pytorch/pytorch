#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorScatterGather.h"
#else

TORCH_CUDA_CU_API void THCTensor_(gather)(
    THCState* state,
    THCTensor* tensor,
    THCTensor* src,
    int dim,
    THCudaLongTensor* index);

#endif
