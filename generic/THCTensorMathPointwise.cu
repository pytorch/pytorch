#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathPointwise.cu"
#else

THC_API void
THCTensor_(cadd)(THCState *state, THCTensor *self_, THCTensor* src1, real value, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    if (value == ScalarConvert<int, real>::to(1)) {
      // self += src2
      if (!THC_pointwiseApply2(state, self_, src2, TensorAddOp<real>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self += value * src2
      if (!THC_pointwiseApply2(state, self_, src2, TensorCAddOp<real>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    if (value == ScalarConvert<int, real>::to(1)) {
      // self = src1 + src2
      if (!THC_pointwiseApply3(state, self_, src1, src2, TensorAddOp<real>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self = src1 + value * src2
      if (!THC_pointwiseApply3(state, self_, src1, src2, TensorCAddOp<real>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(csub)(THCState *state, THCTensor *self_, THCTensor* src1, real value, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    if (value == ScalarConvert<int, real>::to(1)) {
      // self -= src2
      if (!THC_pointwiseApply2(state, self_, src2, TensorSubOp<real>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self += -value * src2
      if (!THC_pointwiseApply2(state, self_, src2,
                                   TensorCAddOp<real>(
                                     ScalarNegate<real>::to(value)))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    if (value == ScalarConvert<int, real>::to(1)) {
      // self = src1 - src2
      if (!THC_pointwiseApply3(state, self_, src1, src2, TensorSubOp<real>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self = src1 - value * src2
      if (!THC_pointwiseApply3(state, self_, src1, src2,
                                   TensorCAddOp<real>(
                                     ScalarNegate<real>::to(value)))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(cmul)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self *= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorMulOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 * src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorMulOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(cpow)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self = pow(self, src2)
    if (!THC_pointwiseApply2(state, self_, src2, TensorCPowOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = pow(src1, src2)
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorCPowOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(cdiv)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self *= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorDivOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 * src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorDivOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

#endif
