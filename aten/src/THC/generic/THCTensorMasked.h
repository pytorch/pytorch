#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMasked.h"
#else

TORCH_CUDA_API void THCTensor_(maskedFill)(THCState *state,
                                    THCTensor *tensor,
                                    THCudaByteTensor *mask,
                                    scalar_t value);


TORCH_CUDA_API void THCTensor_(maskedFillBool)(THCState *state,
                                        THCTensor *tensor,
                                        THCudaBoolTensor *mask,
                                        scalar_t value);

// FIXME: remove now that we have THCudaByteTensor?
TORCH_CUDA_API void THCTensor_(maskedFillByte)(THCState *state,
                                        THCTensor *tensor,
                                        THByteTensor *mask,
                                        scalar_t value);

TORCH_CUDA_API void THCTensor_(maskedCopy)(THCState *state,
                                    THCTensor *tensor,
                                    THCudaByteTensor *mask,
                                    THCTensor *src);

TORCH_CUDA_API void THCTensor_(maskedCopyBool)(THCState *state,
                                        THCTensor *tensor,
                                        THCudaBoolTensor *mask,
                                        THCTensor *src);

// FIXME: remove now that we have THCudaByteTensor?
TORCH_CUDA_API void THCTensor_(maskedCopyByte)(THCState *state,
                                        THCTensor *tensor,
                                        THByteTensor *mask,
                                        THCTensor *src);

#endif
