#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathCompare.h"
#else

THC_API void THCTensor_(ltValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value);
THC_API void THCTensor_(gtValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value);
THC_API void THCTensor_(leValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value);
THC_API void THCTensor_(geValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value);
THC_API void THCTensor_(eqValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value);
THC_API void THCTensor_(neValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value);

THC_API void THCTensor_(ltValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value);
THC_API void THCTensor_(gtValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value);
THC_API void THCTensor_(leValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value);
THC_API void THCTensor_(geValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value);
THC_API void THCTensor_(eqValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value);
THC_API void THCTensor_(neValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value);


#endif
