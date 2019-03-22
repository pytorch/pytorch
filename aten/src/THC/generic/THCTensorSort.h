#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorSort.h"
#else

/* Performs an in-place sort of (keys, values). Only works for slice sizes
   <= 2048 at the moment (slice size == size of keys/values dim `dim`) */
THC_API void THCTensor_(sortKeyValueInplace)(THCState* state,
                                             THCTensor* keys,
                                             THCudaLongTensor* values,
                                             int dim, bool dir);

#endif
