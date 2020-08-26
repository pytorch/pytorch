#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorScatterGather.h"
#else

THC_API void THCTensor_(gather)(THCState* state, THCTensor *tensor, THCTensor *src, int dim, THCudaLongTensor *index);

#endif
