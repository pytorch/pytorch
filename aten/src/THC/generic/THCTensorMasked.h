#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMasked.h"
#else

THC_API void THCTensor_(maskedCopy)(THCState *state,
                                    THCTensor *tensor,
                                    THCudaByteTensor *mask,
                                    THCTensor *src);

THC_API void THCTensor_(maskedCopyBool)(THCState *state,
                                        THCTensor *tensor,
                                        THCudaBoolTensor *mask,
                                        THCTensor *src);

// FIXME: remove now that we have THCudaByteTensor?
THC_API void THCTensor_(maskedCopyByte)(THCState *state,
                                        THCTensor *tensor,
                                        THByteTensor *mask,
                                        THCTensor *src);

#endif
