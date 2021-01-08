#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMasked.h"
#else


TORCH_CUDA_API void THCTensor_(maskedFillBool)(THCState *state,
                                        THCTensor *tensor,
                                        THCudaBoolTensor *mask,
                                        scalar_t value);


TORCH_CUDA_API void THCTensor_(maskedCopyBool)(THCState *state,
                                        THCTensor *tensor,
                                        THCudaBoolTensor *mask,
                                        THCTensor *src);

#endif
