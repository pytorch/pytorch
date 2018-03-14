#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathPairwise.cu"
#else

THC_API void
THCTensor_(add)(THCState *state, THCTensor *self_, THCTensor *src_, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
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
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
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
THCTensor_(add_scaled)(THCState *state, THCTensor *self_, THCTensor *src_, real value, real alpha)
{
#ifdef THC_REAL_IS_HALF
  auto v = THC_half2float(value) * THC_half2float(alpha);
  THCTensor_(add)(state, self_, src_, THC_float2half(v));
#else
  THCTensor_(add)(state, self_, src_, value * alpha);
#endif
}

THC_API void
THCTensor_(sub_scaled)(THCState *state, THCTensor *self_, THCTensor *src_, real value, real alpha)
{
#ifdef THC_REAL_IS_HALF
  auto v = THC_half2float(value) * THC_half2float(alpha);
  THCTensor_(sub)(state, self_, src_, THC_float2half(v));
#else
  THCTensor_(sub)(state, self_, src_, value * alpha);
#endif
}

THC_API void
THCTensor_(mul)(THCState *state, THCTensor *self_, THCTensor *src_, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
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
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
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
THCTensor_(lshift)(THCState* state, THCTensor *self_, THCTensor *src_, real value)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  THCTensor_(mul)(state, self_, src_, pow(2, value));
#elif defined(THC_REAL_IS_HALF)
  return THError("lshift not supported for torch.CudaHalfTensor");
#else
  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, self_, TensorLShiftConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2(state, self_, src_, TensorLShiftConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

THC_API void
THCTensor_(rshift)(THCState* state, THCTensor *self_, THCTensor *src_, real value)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  THCTensor_(mul)(state, self_, src_, pow(2, value));
#elif defined(THC_REAL_IS_HALF)
  return THError("rshift not supported for torch.CudaHalfTensor");
#else
  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, self_, TensorRShiftConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2(state, self_, src_, TensorRShiftConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

THC_API void
THCTensor_(fmod)(THCState *state, THCTensor *self_, THCTensor *src_, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
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
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
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

void THCTensor_(tril)(THCState *state, THCTensor *self_, THCTensor *src_, int64_t k)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
  THArgCheck(src_->nDimension == 2, 1, "expected a matrix");

  if (self_ != src_)
    THCTensor_(resizeAs)(state, self_, src_);

  int64_t stride0 = self_->stride[0];
  int64_t stride1 = self_->stride[1];
  real *start = THCTensor_(data)(state, self_);

  TensorTriOp<real, 0> op(start, stride0, stride1, k);

  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, src_, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2(state, self_, src_, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(triu)(THCState *state, THCTensor *self_, THCTensor *src_, int64_t k)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
  THArgCheck(src_->nDimension == 2, 1, "expected a matrix");

  if (self_ != src_)
    THCTensor_(resizeAs)(state, self_, src_);

  int64_t stride0 = self_->stride[0];
  int64_t stride1 = self_->stride[1];
  real *start = THCTensor_(data)(state, self_);

  TensorTriOp<real, 1> op(start, stride0, stride1, k);

  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, src_, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {

    if (!THC_pointwiseApply2(state, self_, src_, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API int THCTensor_(equal)(THCState *state, THCTensor *self_, THCTensor *src_)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
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

THC_API void
THCTensor_(bitand)(THCState* state, THCTensor *self_, THCTensor *src_, real value)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)
  return THError("bitand only supported for integer type tensors");
#else
  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, self_, TensorBitAndConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2(state, self_, src_, TensorBitAndConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

THC_API void
THCTensor_(bitor)(THCState* state, THCTensor *self_, THCTensor *src_, real value)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)
  return THError("bitor only supported for integer type tensors");
#else
  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, self_, TensorBitOrConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2(state, self_, src_, TensorBitOrConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

THC_API void
THCTensor_(bitxor)(THCState* state, THCTensor *self_, THCTensor *src_, real value)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)
  return THError("bitxor only supported for integer type tensors");
#else
  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, self_, TensorBitXorConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2(state, self_, src_, TensorBitXorConstantOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

#endif
