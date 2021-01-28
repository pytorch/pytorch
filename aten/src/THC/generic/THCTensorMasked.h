#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMasked.h"
#else

TORCH_CUDA_CU_API void THCTensor_(maskedFill)(
    THCState* state,
    THCTensor* tensor,
    THCudaByteTensor* mask,
    scalar_t value);

TORCH_CUDA_CU_API void THCTensor_(maskedFillBool)(
    THCState* state,
    THCTensor* tensor,
    THCudaBoolTensor* mask,
    scalar_t value);

// FIXME: remove now that we have THCudaByteTensor?
TORCH_CUDA_CU_API void THCTensor_(maskedFillByte)(
    THCState* state,
    THCTensor* tensor,
    THByteTensor* mask,
    scalar_t value);

#endif
