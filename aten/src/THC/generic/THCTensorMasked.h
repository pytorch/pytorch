#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMasked.h"
#else

THC_API void THCTensor_(maskedFill)(THCState *state,
                                    THCTensor *tensor,
                                    THCudaByteTensor *mask,
                                    scalar_t value);


THC_API void THCTensor_(maskedFillBool)(THCState *state,
                                        THCTensor *tensor,
                                        THCudaBoolTensor *mask,
                                        scalar_t value);

// FIXME: remove now that we have THCudaByteTensor?
THC_API void THCTensor_(maskedFillByte)(THCState *state,
                                        THCTensor *tensor,
                                        THByteTensor *mask,
                                        scalar_t value);

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

THC_API void THCTensor_(maskedSelect)(THCState *state,
                                      THCTensor *tensor,
                                      THCTensor *src,
                                      THCudaByteTensor *mask);

THC_API void THCTensor_(maskedSelectBool)(THCState *state,
                                          THCTensor *tensor,
                                          THCTensor *src,
                                          THCudaBoolTensor *mask);

// FIXME: remove now that we have THCudaByteTensor?
THC_API void THCTensor_(maskedSelectByte)(THCState *state,
                                          THCTensor *tensor,
                                          THCTensor *src,
                                          THByteTensor *mask);

#endif
