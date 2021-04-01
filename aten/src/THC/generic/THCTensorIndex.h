#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorIndex.h"
#else

TORCH_CUDA_CU_API void THCTensor_(indexSelect)(
    THCState* state,
    THCTensor* tensor,
    THCTensor* src,
    int dim,
    THCudaLongTensor* index);
TORCH_CUDA_CU_API void THCTensor_(take)(
    THCState* state,
    THCTensor* res_,
    THCTensor* src,
    THCudaLongTensor* index);
TORCH_CUDA_CU_API void THCTensor_(put)(
    THCState* state,
    THCTensor* res_,
    THCudaLongTensor* indices,
    THCTensor* src,
    int accumulate);

#endif
