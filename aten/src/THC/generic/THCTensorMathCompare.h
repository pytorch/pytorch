#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathCompare.h"
#else

THC_API void THCTensor_(ltValue)(THCState *state, THCudaBoolTensor *self_, THCTensor *src, scalar_t value);
THC_API void THCTensor_(gtValue)(THCState *state, THCudaBoolTensor *self_, THCTensor *src, scalar_t value);
THC_API void THCTensor_(leValue)(THCState *state, THCudaBoolTensor *self_, THCTensor *src, scalar_t value);
THC_API void THCTensor_(geValue)(THCState *state, THCudaBoolTensor *self_, THCTensor *src, scalar_t value);
THC_API void THCTensor_(eqValue)(THCState *state, THCudaBoolTensor *self_, THCTensor *src, scalar_t value);
THC_API void THCTensor_(neValue)(THCState *state, THCudaBoolTensor *self_, THCTensor *src, scalar_t value);

THC_API void THCTensor_(ltValueT)(THCState *state, THCTensor *self_, THCTensor *src, scalar_t value);
THC_API void THCTensor_(gtValueT)(THCState *state, THCTensor *self_, THCTensor *src, scalar_t value);
THC_API void THCTensor_(leValueT)(THCState *state, THCTensor *self_, THCTensor *src, scalar_t value);
THC_API void THCTensor_(geValueT)(THCState *state, THCTensor *self_, THCTensor *src, scalar_t value);
THC_API void THCTensor_(eqValueT)(THCState *state, THCTensor *self_, THCTensor *src, scalar_t value);
THC_API void THCTensor_(neValueT)(THCState *state, THCTensor *self_, THCTensor *src, scalar_t value);

THC_API void THCTensor_(ltValueByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src, scalar_t value);
THC_API void THCTensor_(gtValueByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src, scalar_t value);
THC_API void THCTensor_(leValueByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src, scalar_t value);
THC_API void THCTensor_(geValueByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src, scalar_t value);
THC_API void THCTensor_(eqValueByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src, scalar_t value);
THC_API void THCTensor_(neValueByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src, scalar_t value);

#endif
