#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathPairwise.cu"
#else

#include <ATen/NamedTensorUtils.h>

static int THCTensor_(equalImpl)(THCState *state, THCTensor *self_, THCTensor *src_)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
  if (!THCTensor_(isSameSizeAs(state, self_, src_))) {
    return 0;
  }

  // This is not as efficient as TH, but the basic idea: create a buffer that stores
  // 1 if the two tensors are equal at a position, otherwise 0. If the minimum value
  // in this buffer is 1, the two tensors are equal, otherwise they are not

  // Both tensors are empty
  if(THTensor_(nElement)(self_) == 0) return true;

  THCudaByteTensor *buf = THCudaByteTensor_newWithSize(state, self_->sizes(), {});

  if (!THC_pointwiseApply3<uint8_t, scalar_t, scalar_t>(state, buf, self_, src_, TensorEQOp<scalar_t, unsigned char>())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  unsigned char min = THCudaByteTensor_minall(state, buf);

  THCudaByteTensor_free(state, buf);

  return min != 0;
}

int THCTensor_(equal)(THCState *state, THCTensor *self_, THCTensor *src_) {
#ifdef BUILD_NAMEDTENSOR
  if (!at::namedinference::are_names_equal(self_, src_)) {
    return 0;
  }
  at::NoNamesGuard guard;
#endif
  return THCTensor_(equalImpl)(state, self_, src_);
}

void THCTensor_(bitand)(THCState* state, THCTensor *self_, THCTensor *src_, scalar_t value)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)
  return THError("bitand only supported for integer type tensors");
#else
  if (self_ == src_) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorBitAndConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src_, TensorBitAndConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

void THCTensor_(bitor)(THCState* state, THCTensor *self_, THCTensor *src_, scalar_t value)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)
  return THError("bitor only supported for integer type tensors");
#else
  if (self_ == src_) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorBitOrConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src_, TensorBitOrConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

void THCTensor_(bitxor)(THCState* state, THCTensor *self_, THCTensor *src_, scalar_t value)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)
  return THError("bitxor only supported for integer type tensors");
#else
  if (self_ == src_) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorBitXorConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src_, TensorBitXorConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

#if !defined(THC_REAL_IS_BOOL)

void THCTensor_(mul)(THCState *state, THCTensor *self_, THCTensor *src_, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorMulConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src_, TensorMulConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(div)(THCState* state, THCTensor *self_, THCTensor *src_, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
  THArgCheck(value != ScalarConvert<int, scalar_t>::to(0), 3, "divide by zero");

  if (self_ == src_) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorDivConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src_, TensorDivConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(lshift)(THCState* state, THCTensor *self_, THCTensor *src_, scalar_t value)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  THCTensor_(mul)(state, self_, src_, pow(2, value));
#elif defined(THC_REAL_IS_HALF)
  return THError("lshift not supported for torch.CudaHalfTensor");
#else
  if (self_ == src_) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorLShiftConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src_, TensorLShiftConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

void THCTensor_(rshift)(THCState* state, THCTensor *self_, THCTensor *src_, scalar_t value)
{
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  THCTensor_(mul)(state, self_, src_, pow(2, -value));
#elif defined(THC_REAL_IS_HALF)
  return THError("rshift not supported for torch.CudaHalfTensor");
#else
  if (self_ == src_) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorRShiftConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src_, TensorRShiftConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

void THCTensor_(fmod)(THCState *state, THCTensor *self_, THCTensor *src_, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorFmodOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src_, TensorFmodOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(remainder)(THCState *state, THCTensor *self_, THCTensor *src_, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorRemainderOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src_, TensorRemainderOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(triu)(THCState *state, THCTensor *self_, THCTensor *src_, int64_t k)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
  THArgCheck(!src_->is_empty() && src_->dim() == 2, 1, "expected a matrix");

  if (self_ != src_)
    THCTensor_(resizeAs)(state, self_, src_);

  int64_t stride0 = self_->stride(0);
  int64_t stride1 = self_->stride(1);
  scalar_t *start = THCTensor_(data)(state, self_);

  TensorTriOp<scalar_t, 1> op(start, stride0, stride1, k);

  if (self_ == src_) {
    if (!THC_pointwiseApply1<scalar_t>(state, src_, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src_, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

#endif

#endif
