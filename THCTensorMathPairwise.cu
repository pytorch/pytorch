#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

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
