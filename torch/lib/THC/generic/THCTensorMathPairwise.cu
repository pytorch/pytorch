#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathPairwise.cu"
#else

THC_API void
THCTensor_(add)(THCState *state, THCTensor *self_, THCTensor *src_, real value)
{
  THAssert(THCTensor_(checkGPU)(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, self_, TensorAddConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2(state, self_, src_, TensorAddConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(sub)(THCState *state, THCTensor *self_, THCTensor *src_, real value)
{
  THCTensor_(add)(state, self_, src_, ScalarNegate<real>::to(value));
}

THC_API void
THCTensor_(mul)(THCState *state, THCTensor *self_, THCTensor *src_, real value)
{
  THAssert(THCTensor_(checkGPU)(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, self_, TensorMulConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2(state, self_, src_, TensorMulConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(div)(THCState* state, THCTensor *self_, THCTensor *src_, real value)
{
  THAssert(THCTensor_(checkGPU)(state, 2, self_, src_));
  THArgCheck(value != ScalarConvert<int, real>::to(0), 3, "divide by zero");

  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, self_,
                                 TensorMulConstantOp<real>(
                                   ScalarInv<real>::to(value)))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2(state, self_, src_,
                                 TensorMulConstantOp<real>(
                                   ScalarInv<real>::to(value)))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

#endif
