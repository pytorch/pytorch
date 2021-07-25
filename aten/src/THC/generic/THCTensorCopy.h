#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorCopy.h"
#else

TORCH_CUDA_CU_API void THCTensor_(
    copy)(THCState* state, THCTensor* self, THCTensor* src);
TORCH_CUDA_CU_API void THCTensor_(
    copyIgnoringOverlaps)(THCState* state, THCTensor* self, THCTensor* src);

TORCH_CUDA_CU_API void THCTensor_(
    copyAsyncCPU)(THCState* state, THCTensor* self, THTensor* src);
TORCH_CUDA_CU_API void THTensor_(
    copyAsyncCuda)(THCState* state, THTensor* self, THCTensor* src);

#endif
