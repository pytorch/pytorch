#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMode.h"
#else

/* Returns the mode, and index of the mode, for the set of values
 * along a given dimension in the input tensor. */
TORCH_CUDA_CU_API void THCTensor_(mode)(
    THCState* state,
    THCTensor* values,
    THCudaLongTensor* indices,
    THCTensor* input,
    int dimension,
    int keepdim);

#endif // THC_GENERIC_FILE
