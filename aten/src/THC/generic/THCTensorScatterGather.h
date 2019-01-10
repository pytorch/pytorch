#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorScatterGather.h"
#else

THC_API void THCTensor_(gather)(THCState* state, THCTensor *tensor, THCTensor *src, int dim, THCudaLongTensor *index);
THC_API void THCTensor_(scatter)(THCState* state, THCTensor *tensor, int dim, THCudaLongTensor *index, THCTensor *src);
THC_API void THCTensor_(scatterAdd)(THCState* state, THCTensor *tensor, int dim, THCudaLongTensor *index, THCTensor *src);
THC_API void THCTensor_(scatterFill)(THCState* state, THCTensor *tensor, int dim, THCudaLongTensor *index, real value);

#endif
