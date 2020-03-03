#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorIndex.h"
#else

THC_API void THCTensor_(indexCopy)(THCState *state, THCTensor *res_, int dim, THCudaLongTensor *indices, THCTensor *src);
THC_API void THCTensor_(indexFill)(THCState *state, THCTensor *tensor, int dim, THCudaLongTensor *index, scalar_t val);
THC_API void THCTensor_(indexSelect)(THCState *state, THCTensor *tensor, THCTensor *src, int dim, THCudaLongTensor *index);
THC_API void THCTensor_(take)(THCState *state, THCTensor *res_, THCTensor *src, THCudaLongTensor *index);
THC_API void THCTensor_(put)(THCState *state, THCTensor *res_, THCudaLongTensor *indices, THCTensor *src, int accumulate);

#endif
