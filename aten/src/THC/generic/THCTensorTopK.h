#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorTopK.h"
#else

/* Returns the set of all kth smallest (or largest) elements, depending */
/* on `dir` */
THC_API void THCTensor_(topk)(THCState* state,
                               THCTensor* topK,
                               THCudaLongTensor* indices,
                               THCTensor* input,
                               int64_t k, int dim, int dir, int sorted);

#endif // THC_GENERIC_FILE
