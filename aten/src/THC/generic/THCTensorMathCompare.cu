#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathCompare.cu"
#else

THC_API void THCTensor_(ltValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue(state, self_, src,
                   TensorLTValueOp<typename TensorUtils<THCTensor>::DataType,
                   unsigned char>(value));
}

THC_API void THCTensor_(gtValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue(state, self_, src,
                   TensorGTValueOp<typename TensorUtils<THCTensor>::DataType,
                   unsigned char>(value));
}

THC_API void THCTensor_(leValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue(state, self_, src,
                   TensorLEValueOp<typename TensorUtils<THCTensor>::DataType,
                   unsigned char>(value));
}

THC_API void THCTensor_(geValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue(state, self_, src,
                   TensorGEValueOp<typename TensorUtils<THCTensor>::DataType,
                   unsigned char>(value));
}

THC_API void THCTensor_(eqValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue(state, self_, src,
                   TensorEQValueOp<typename TensorUtils<THCTensor>::DataType,
                   unsigned char>(value));
}

THC_API void THCTensor_(neValue)(THCState *state, THCudaByteTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue(state, self_, src,
                   TensorNEValueOp<typename TensorUtils<THCTensor>::DataType,
                   unsigned char>(value));
}

THC_API void THCTensor_(ltValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue(state, self_, src,
                   TensorLTValueOp<typename TensorUtils<THCTensor>::DataType,
                   typename TensorUtils<THCTensor>::DataType>(value));
}

THC_API void THCTensor_(gtValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue(state, self_, src,
                   TensorGTValueOp<typename TensorUtils<THCTensor>::DataType,
                   typename TensorUtils<THCTensor>::DataType>(value));
}

THC_API void THCTensor_(leValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue(state, self_, src,
                   TensorLEValueOp<typename TensorUtils<THCTensor>::DataType,
                   typename TensorUtils<THCTensor>::DataType>(value));
}

THC_API void THCTensor_(geValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue(state, self_, src,
                   TensorGEValueOp<typename TensorUtils<THCTensor>::DataType,
                   typename TensorUtils<THCTensor>::DataType>(value));
}

THC_API void THCTensor_(eqValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue(state, self_, src,
                   TensorEQValueOp<typename TensorUtils<THCTensor>::DataType,
                   typename TensorUtils<THCTensor>::DataType>(value));
}

THC_API void THCTensor_(neValueT)(THCState *state, THCTensor *self_, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THC_logicalValue(state, self_, src,
                   TensorNEValueOp<typename TensorUtils<THCTensor>::DataType,
                   typename TensorUtils<THCTensor>::DataType>(value));
}

#endif
