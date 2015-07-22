#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"

struct TensorAddConstantOp {
  TensorAddConstantOp(float v) : val(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = *in + val;
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v += val;
  }

  const float val;
};

void THCudaTensor_add(THCState *state, THCudaTensor *self_, THCudaTensor *src_, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THCudaTensor_pointwiseApply1(state, self_, TensorAddConstantOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src_);

    if (!THCudaTensor_pointwiseApply2(state, self_, src_, TensorAddConstantOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorMulConstantOp {
  TensorMulConstantOp(float v) : val(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = *in * val;
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v *= val;
  }

  const float val;
};

void THCudaTensor_mul(THCState *state, THCudaTensor *self_, THCudaTensor *src_, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THCudaTensor_pointwiseApply1(state, self_, TensorMulConstantOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src_);

    if (!THCudaTensor_pointwiseApply2(state, self_, src_, TensorMulConstantOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCudaTensor_div(THCState* state, THCudaTensor *self_, THCudaTensor *src_, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src_));
  THArgCheck(value != 0.0f, 3, "divide by zero");

  if (self_ == src_) {
    if (!THCudaTensor_pointwiseApply1(state, self_, TensorMulConstantOp(1.0f / value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src_);

    if (!THCudaTensor_pointwiseApply2(state, self_, src_, TensorMulConstantOp(1.0f / value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

template <int Upper>
struct TensorTriOp {
  TensorTriOp(float *start_, long stride0_, long stride1_, long k_)
    : start(start_), stride0(stride0_), stride1(stride1_), k(k_) {}

  __device__ __forceinline__ int mask(float *in) {
    ptrdiff_t n = in - start;
    long row, col;
    if (stride0 > stride1)
    {
      row = (long) (n / stride0);
      col = (long) ((n % stride0) / stride1);
    }
    else
    {
      row = (long) ((n % stride1) / stride0);
      col = (long) (n / stride1);
    }

    return Upper ? (col - row >= k) : (col - row <= k);
  }

  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = mask(in) ? *in : 0;
  }

  __device__ __forceinline__ void operator()(float* v) {
    if (!mask(v))
      *v = 0;
  }

  const float *start;
  const long stride0, stride1, k;
};

void THCudaTensor_tril(THCState *state, THCudaTensor *self_, THCudaTensor *src_, long k)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src_));
  THArgCheck(src_->nDimension == 2, 1, "expected a matrix");

  THCudaTensor *src = src_;
  if (self_ == src_)
    src = THCudaTensor_newContiguous(state, src_);

  long stride0 = src->stride[0];
  long stride1 = src->stride[1];
  float *start = THCudaTensor_data(state, src) + src->storageOffset;

  TensorTriOp<0> op(start, stride0, stride1, k);

  if (self_ == src_) {
    if (!THCudaTensor_pointwiseApply1(state, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THCudaTensor_pointwiseApply2(state, self_, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  if (self_ == src_)
    THCudaTensor_freeCopyTo(state, src, src_);

  THCudaCheck(cudaGetLastError());
}

void THCudaTensor_triu(THCState *state, THCudaTensor *self_, THCudaTensor *src_, long k)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src_));
  THArgCheck(src_->nDimension == 2, 1, "expected a matrix");

  THCudaTensor *src = src_;
  if (self_ == src_)
    src = THCudaTensor_newContiguous(state, src_);

  long stride0 = src->stride[0];
  long stride1 = src->stride[1];
  float *start = THCudaTensor_data(state, src) + src->storageOffset;

  TensorTriOp<1> op(start, stride0, stride1, k);

  if (self_ == src_) {
    if (!THCudaTensor_pointwiseApply1(state, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THCudaTensor_pointwiseApply2(state, self_, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  if (self_ == src_)
    THCudaTensor_freeCopyTo(state, src, src_);

  THCudaCheck(cudaGetLastError());
}
