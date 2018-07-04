#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathCompare.cu"
#else

THC_API void THCTensor_(ltValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<uint8_t, real>(state, self_, src,
                                  TensorLTValueOp<real,
                                  unsigned char>(value));
}

THC_API void THCTensor_(gtValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<uint8_t, real>(state, self_, src,
                                  TensorGTValueOp<real,
                                  unsigned char>(value));
}

THC_API void THCTensor_(leValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<uint8_t, real>(state, self_, src,
                                  TensorLEValueOp<real,
                                  unsigned char>(value));
}

THC_API void THCTensor_(geValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<uint8_t, real>(state, self_, src,
                                  TensorGEValueOp<real,
                                  unsigned char>(value));
}

THC_API void THCTensor_(eqValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<uint8_t, real>(state, self_, src,
                                  TensorEQValueOp<real,
                                  unsigned char>(value));
}

THC_API void THCTensor_(neValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<uint8_t, real>(state, self_, src,
                                  TensorNEValueOp<real,
                                  unsigned char>(value));
}

THC_API void THCTensor_(ltValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<real, real>(state, self_, src,
                                  TensorLTValueOp<real,
                                  real>(value));
}

THC_API void THCTensor_(gtValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<real, real>(state, self_, src,
                               TensorGTValueOp<real,
                              real>(value));
}

THC_API void THCTensor_(leValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<real, real>(state, self_, src,
                               TensorLEValueOp<real,
                               real>(value));
}

THC_API void THCTensor_(geValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<real, real>(state, self_, src,
                               TensorGEValueOp<real,
                               real>(value));
}

THC_API void THCTensor_(eqValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<real, real>(state, self_, src,
                               TensorEQValueOp<real,
                               real>(value));
}

THC_API void THCTensor_(neValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<real, real>(state, self_, src,
                              TensorNEValueOp<real,
                              real>(value));
}

#endif
