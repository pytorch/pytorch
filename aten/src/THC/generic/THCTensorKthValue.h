#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorKthValue.h"
#else

/* Returns the kth smallest element */
THC_API void THCTensor_(kthvalue)(THCState* state,
                                  THCTensor* kthValue,
                                  THCudaLongTensor* indices,
                                  THCTensor* input,
                                  int64_t k, int dim, int keepDim);

#endif // THC_GENERIC_FILE
