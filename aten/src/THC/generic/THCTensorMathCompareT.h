#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathCompareT.h"
#else

THC_API void THCTensor_(ltTensor)(THCState *state, THCudaBoolTensor *self_, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(gtTensor)(THCState *state, THCudaBoolTensor *self_, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(leTensor)(THCState *state, THCudaBoolTensor *self_, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(geTensor)(THCState *state, THCudaBoolTensor *self_, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(eqTensor)(THCState *state, THCudaBoolTensor *self_, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(neTensor)(THCState *state, THCudaBoolTensor *self_, THCTensor *src1, THCTensor *src2);

THC_API void THCTensor_(ltTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(gtTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(leTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(geTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(eqTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(neTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2);

THC_API void THCTensor_(ltTensorByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(gtTensorByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(leTensorByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(geTensorByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(eqTensorByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2);
THC_API void THCTensor_(neTensorByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2);

#endif
