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
  THAssert(THCTensor_(checkGPU)(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, self_, TensorSubConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2(state, self_, src_, TensorSubConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
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
    if (!THC_pointwiseApply1(state, self_, TensorDivConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2(state, self_, src_, TensorDivConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(fmod)(THCState *state, THCTensor *self_, THCTensor *src_, real value)
{
  THAssert(THCTensor_(checkGPU)(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, self_, TensorFmodOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2(state, self_, src_, TensorFmodOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(remainder)(THCState *state, THCTensor *self_, THCTensor *src_, real value)
{
  THAssert(THCTensor_(checkGPU)(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, self_, TensorRemainderOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2(state, self_, src_, TensorRemainderOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(tril)(THCState *state, THCTensor *self_, THCTensor *src_, long k)
{
  THAssert(THCTensor_(checkGPU)(state, 2, self_, src_));
  THArgCheck(src_->nDimension == 2, 1, "expected a matrix");

  THCTensor *src = src_;
  if (self_ == src_)
    src = THCTensor_(newContiguous)(state, src_);

  long stride0 = src->stride[0];
  long stride1 = src->stride[1];
  real *start = THCTensor_(data)(state, src) + src->storageOffset;

  TensorTriOp<real, 0> op(start, stride0, stride1, k);

  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  if (self_ == src_)
    THCTensor_(freeCopyTo)(state, src, src_);

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(triu)(THCState *state, THCTensor *self_, THCTensor *src_, long k)
{
  THAssert(THCTensor_(checkGPU)(state, 2, self_, src_));
  THArgCheck(src_->nDimension == 2, 1, "expected a matrix");

  THCTensor *src = src_;
  if (self_ == src_)
    src = THCTensor_(newContiguous)(state, src_);

  long stride0 = src->stride[0];
  long stride1 = src->stride[1];
  real *start = THCTensor_(data)(state, src) + src->storageOffset;

  TensorTriOp<real, 1> op(start, stride0, stride1, k);

  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  if (self_ == src_)
    THCTensor_(freeCopyTo)(state, src, src_);

  THCudaCheck(cudaGetLastError());
}

THC_API int THCTensor_(equal)(THCState *state, THCTensor *self_, THCTensor *src_)
{
  THAssert(THCTensor_(checkGPU)(state, 2, self_, src_));
  if (!THCTensor_(isSameSizeAs(state, self_, src_))) {
    return 0;
  }

  // This is not as efficient as TH, but the basic idea: create a buffer that stores
  // 1 if the two tensors are equal at a position, otherwise 0. If the minimum value
  // in this buffer is 1, the two tensors are equal, otherwise they are not

  THLongStorage *size = THCTensor_(newSizeOf)(state, self_);
  THCudaByteTensor *buf = THCudaByteTensor_newWithSize(state, size, NULL);

  if (!THC_pointwiseApply3(state, buf, self_, src_, TensorEQOp<real, unsigned char>())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  unsigned char min = THCudaByteTensor_minall(state, buf);

  THLongStorage_free(size);
  THCudaByteTensor_free(state, buf);

  return min != 0;
}

#endif
