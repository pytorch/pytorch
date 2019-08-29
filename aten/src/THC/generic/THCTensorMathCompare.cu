#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathCompare.cu"
#else

void THCTensor_(ltValue)(THCState *state, THCudaBoolTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<bool, scalar_t>(state, self_, src,
                                  TensorLTValueOp<scalar_t,
                                  bool>(value));
}

void THCTensor_(gtValue)(THCState *state, THCudaBoolTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<bool, scalar_t>(state, self_, src,
                                  TensorGTValueOp<scalar_t,
                                  bool>(value));
}

void THCTensor_(leValue)(THCState *state, THCudaBoolTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<bool, scalar_t>(state, self_, src,
                                  TensorLEValueOp<scalar_t,
                                  bool>(value));
}

void THCTensor_(geValue)(THCState *state, THCudaBoolTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<bool, scalar_t>(state, self_, src,
                                  TensorGEValueOp<scalar_t,
                                  bool>(value));
}

void THCTensor_(eqValue)(THCState *state, THCudaBoolTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<bool, scalar_t>(state, self_, src,
                                  TensorEQValueOp<scalar_t,
                                  bool>(value));
}

void THCTensor_(neValue)(THCState *state, THCudaBoolTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<bool, scalar_t>(state, self_, src,
                                  TensorNEValueOp<scalar_t,
                                  bool>(value));
}

void THCTensor_(ltValueT)(THCState *state, THCTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<scalar_t, scalar_t>(state, self_, src,
                                  TensorLTValueOp<scalar_t,
                                  scalar_t>(value));
}

void THCTensor_(gtValueT)(THCState *state, THCTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<scalar_t, scalar_t>(state, self_, src,
                               TensorGTValueOp<scalar_t,
                              scalar_t>(value));
}

void THCTensor_(leValueT)(THCState *state, THCTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<scalar_t, scalar_t>(state, self_, src,
                               TensorLEValueOp<scalar_t,
                               scalar_t>(value));
}

void THCTensor_(geValueT)(THCState *state, THCTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<scalar_t, scalar_t>(state, self_, src,
                               TensorGEValueOp<scalar_t,
                               scalar_t>(value));
}

void THCTensor_(eqValueT)(THCState *state, THCTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<scalar_t, scalar_t>(state, self_, src,
                               TensorEQValueOp<scalar_t,
                               scalar_t>(value));
}

void THCTensor_(neValueT)(THCState *state, THCTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<scalar_t, scalar_t>(state, self_, src,
                              TensorNEValueOp<scalar_t,
                              scalar_t>(value));
}


void THCTensor_(ltValueByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<unsigned char, scalar_t>(state, self_, src,
                                  TensorLTValueOp<scalar_t,
                                  unsigned char>(value));
}

void THCTensor_(gtValueByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<unsigned char, scalar_t>(state, self_, src,
                                  TensorGTValueOp<scalar_t,
                                  unsigned char>(value));
}

void THCTensor_(leValueByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<unsigned char, scalar_t>(state, self_, src,
                                  TensorLEValueOp<scalar_t,
                                  unsigned char>(value));
}

void THCTensor_(geValueByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<unsigned char, scalar_t>(state, self_, src,
                                  TensorGEValueOp<scalar_t,
                                  unsigned char>(value));
}

void THCTensor_(eqValueByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<unsigned char, scalar_t>(state, self_, src,
                                  TensorEQValueOp<scalar_t,
                                  unsigned char>(value));
}

void THCTensor_(neValueByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue<unsigned char, scalar_t>(state, self_, src,
                                  TensorNEValueOp<scalar_t,
                                  unsigned char>(value));
}

#endif
