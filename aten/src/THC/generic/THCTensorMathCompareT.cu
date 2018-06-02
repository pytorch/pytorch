#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathCompareT.cu"
#else

THC_API void
THCTensor_(ltTensor)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<uint8_t, real>(state, self_, src1, src2,
                                   TensorLTOp<real,
                                   unsigned char>());
}

THC_API void
THCTensor_(gtTensor)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<uint8_t, real>(state, self_, src1, src2,
                                   TensorGTOp<real,
                                   unsigned char>());
}

THC_API void
THCTensor_(leTensor)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<uint8_t, real>(state, self_, src1, src2,
                                   TensorLEOp<real,
                                   unsigned char>());
}

THC_API void
THCTensor_(geTensor)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<uint8_t, real>(state, self_, src1, src2,
                                   TensorGEOp<real,
                                   unsigned char>());
}

THC_API void
THCTensor_(eqTensor)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<uint8_t, real>(state, self_, src1, src2,
                                   TensorEQOp<real,
                                   unsigned char>());
}

THC_API void
THCTensor_(neTensor)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<uint8_t, real>(state, self_, src1, src2,
                                   TensorNEOp<real,
                                   unsigned char>());
}

THC_API void
THCTensor_(ltTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<real, real>(state, self_, src1, src2,
                                TensorLTOp<real,
                                real>());
}

THC_API void
THCTensor_(gtTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<real, real>(state, self_, src1, src2,
                                TensorGTOp<real,
                                real>());
}

THC_API void
THCTensor_(leTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<real, real>(state, self_, src1, src2,
                                TensorLEOp<real,
                                real>());
}

THC_API void
THCTensor_(geTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<real, real>(state, self_, src1, src2,
                                TensorGEOp<real,
                                real>());
}

THC_API void
THCTensor_(eqTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<real, real>(state, self_, src1, src2,
                                TensorEQOp<real,
                                real>());
}

THC_API void
THCTensor_(neTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<real, real>(state, self_, src1, src2,
                                TensorNEOp<real,
                                real>());
}

#endif
