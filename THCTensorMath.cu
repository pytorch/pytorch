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

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

struct TensorFillOp {
  TensorFillOp(float v) : val(v) {}
  __device__ __forceinline__ void operator()(float* v) { *v = val; }

  const float val;
};

void THCudaTensor_fill(THCState* state, THCudaTensor *self_, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  if (!THCudaTensor_pointwiseApply1(state, self_, TensorFillOp(value))) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

void THCudaTensor_zero(THCState *state, THCudaTensor *self_)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  if (THCudaTensor_isContiguous(state, self_)) {
    THCudaCheck(cudaMemsetAsync(THCudaTensor_data(state, self_),
                                0,
                                sizeof(float) * THCudaTensor_nElement(state, self_),
                                THCState_getCurrentStream(state)));
  } else {
    if (!THCudaTensor_pointwiseApply1(state, self_, TensorFillOp(0))) {
      THArgCheck(false, 1, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCudaTensor_zeros(THCState *state, THCudaTensor *r_, THLongStorage *size)
{
  THAssert(THCudaTensor_checkGPU(state, 1, r_));
  THCudaTensor_resize(state, r_, size, NULL);
  THCudaTensor_zero(state, r_);
}

void THCudaTensor_ones(THCState *state, THCudaTensor *r_, THLongStorage *size)
{
  THAssert(THCudaTensor_checkGPU(state, 1, r_));
  THCudaTensor_resize(state, r_, size, NULL);
  THCudaTensor_fill(state, r_, 1);
}

void THCudaTensor_reshape(THCState *state, THCudaTensor *r_, THCudaTensor *t, THLongStorage *size)
{
  THAssert(THCudaTensor_checkGPU(state, 2, r_, t));
  THCudaTensor_resize(state, r_, size, NULL);
  THCudaTensor_copy(state, r_, t);
}

long THCudaTensor_numel(THCState *state, THCudaTensor *t)
{
  return THCudaTensor_nElement(state, t);
}

struct TensorCPowOp {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = powf(*out, *in);
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = powf(*in1, *in2);
  }
};

void THCudaTensor_cpow(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THArgCheck(THCudaTensor_nElement(state, src1) ==
             THCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self = pow(self, src2)
    if (!THCudaTensor_pointwiseApply2(state, self_, src2, TensorCPowOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src1);

    // self = pow(src1, src2)
    if (!THCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorCPowOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorDivOp {
  __device__ __forceinline__ void
  operator()(float* out, float* in) {
    *out /= *in;
  }

  __device__ __forceinline__ void
  operator()(float* out, float* in1, float* in2) {
    *out = *in1 / *in2;
  }
};

void THCudaTensor_cdiv(THCState* state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THArgCheck(THCudaTensor_nElement(state, src1) ==
             THCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self *= src2
    if (!THCudaTensor_pointwiseApply2(state, self_, src2, TensorDivOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src1);

    // self = src1 * src2
    if (!THCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorDivOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorAddCMulOp {
  TensorAddCMulOp(float v) : val(v) {}

  __device__ __forceinline__ void
  operator()(float* out, float* in1, float* in2) {
    *out += val * *in1 * *in2;
  }

  float val;
};

void THCudaTensor_addcmul(THCState *state, THCudaTensor *self_, THCudaTensor *t, float value, THCudaTensor *src1, THCudaTensor *src2)
{
  THAssert(THCudaTensor_checkGPU(state, 4, self_, t, src1, src2));
  if(self_ != t)
  {
    THCudaTensor_resizeAs(state, self_, t);
    THCudaTensor_copy(state, self_, t);
  }
  THCudaTensor_resizeAs(state, self_, src1);

  THArgCheck(THCudaTensor_nElement(state, src1) ==
             THCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (!THCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorAddCMulOp(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorAddCDivOp {
  TensorAddCDivOp(float v) : val(v) {}

  __device__ __forceinline__ void
  operator()(float* out, float* in1, float* in2) {
    *out += val * *in1 / *in2;
  }

  float val;
};

void THCudaTensor_addcdiv(THCState *state, THCudaTensor *self_, THCudaTensor *t, float value, THCudaTensor *src1, THCudaTensor *src2)
{
  THAssert(THCudaTensor_checkGPU(state, 4, self_, t, src1, src2));
  if(self_ != t)
  {
    THCudaTensor_resizeAs(state, self_, t);
    THCudaTensor_copy(state, self_, t);
  }

  THCudaTensor_resizeAs(state, self_, src1);
  THArgCheck(THCudaTensor_nElement(state, src1) == THCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (!THCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorAddCDivOp(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

float THCudaTensor_minall(THCState *state, THCudaTensor *self)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self));
  self = THCudaTensor_newContiguous(state, self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(state, self));

  float result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(state, self), (float)(THInf), thrust::minimum<float>());

  THCudaTensor_free(state, self);
  return result;
}

float THCudaTensor_maxall(THCState *state, THCudaTensor *self)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self));
  self = THCudaTensor_newContiguous(state, self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(state, self));

  float result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(state, self), (float)(-THInf), thrust::maximum<float>());

  THCudaTensor_free(state, self);
  return result;
}

float THCudaTensor_sumall(THCState *state, THCudaTensor *self)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self));
  self = THCudaTensor_newContiguous(state, self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(state, self));

  float result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(state, self), (float)(0), thrust::plus<float>());

  THCudaTensor_free(state, self);
  return result;
}

float THCudaTensor_prodall(THCState *state, THCudaTensor *self)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self));
  self = THCudaTensor_newContiguous(state, self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(state, self));

  float result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(state, self), (float)(1), thrust::multiplies<float>());

  THCudaTensor_free(state, self);
  return result;
}

struct dim4 {
    unsigned arr[4];

    __host__ dim4(unsigned init=0) {
        for(unsigned i=0; i<4; i++) { arr[i] = init; }
    }

    __host__ __device__ unsigned& operator[](const unsigned& idx) { return arr[idx]; }
};

void THCudaTensor_sum(THCState* state, THCudaTensor *self, THCudaTensor *src, long dimension)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self, src));
  THCudaTensor_reduceDim(
    state, self, src,
    thrust::identity<float>(), thrust::plus<float>(), 0.0f, dimension);

  THCudaCheck(cudaGetLastError());
}

void THCudaTensor_prod(THCState* state, THCudaTensor *self, THCudaTensor *src, long dimension)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self, src));
  THCudaTensor_reduceDim(
    state, self, src,
    thrust::identity<float>(), thrust::multiplies<float>(), 1.0f, dimension);

  THCudaCheck(cudaGetLastError());
}

struct logicalall_functor
{
  __host__ __device__ float operator()(const float& x, const float& y) const
  {
    return x && y;
  }
};

struct logicalany_functor
{
  __host__ __device__ float operator()(const float& x, const float& y) const
  {
    return x || y;
  }
};

int THCudaTensor_logicalall(THCState *state, THCudaTensor *self) {
  THAssert(THCudaTensor_checkGPU(state, 1, self));
  self = THCudaTensor_newContiguous(state, self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(state, self));

  int result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(state, self), (float)(1), logicalall_functor());

  THCudaTensor_free(state, self);
  return result;
}

int THCudaTensor_logicalany(THCState *state, THCudaTensor *self) {
  THAssert(THCudaTensor_checkGPU(state, 1, self));
  self = THCudaTensor_newContiguous(state, self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(state, self));

  int result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(state, self), (float)(0), logicalany_functor());

  THCudaTensor_free(state, self);
  return result;
}
