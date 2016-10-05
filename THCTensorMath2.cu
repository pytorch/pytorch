#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"
#include "THCTensorMathReduce.cuh"
#include "THCTensorMathPointwise.cuh"

#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

struct TensorTPowOp {
  TensorTPowOp(float v) : val(v) {}

  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = powf(val, *in);
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v = powf(val, *v);
  }

  const float val;
};

void THCudaTensor_tpow(THCState *state, THCudaTensor *self_, float value, THCudaTensor *src)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1(state, self_, TensorTPowOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, TensorTPowOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorATan2Op {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = atan2f(*a, *b);
  }
};

void THCudaTensor_atan2(THCState *state, THCudaTensor *self_, THCudaTensor *tx, THCudaTensor *ty)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self_, tx, ty));
  THArgCheck(THCudaTensor_nElement(state, tx) ==
             THCudaTensor_nElement(state, ty), 3, "sizes do not match");
  THCudaTensor_resizeAs(state, self_, tx);

  if (!THC_pointwiseApply3(state, self_, tx, ty, TensorATan2Op())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorClampOp {
  TensorClampOp(float min, float max) : minValue(min), maxValue(max) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = max(min(*in, maxValue), minValue);
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v = max(min(*v, maxValue), minValue);
  }

  const float minValue;
  const float maxValue;
};

void THCudaTensor_clamp(THCState *state, THCudaTensor *self_, THCudaTensor *src, float min_value,
  float max_value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1(state, self_, TensorClampOp(min_value, max_value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, TensorClampOp(min_value, max_value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorLerpOp {
  TensorLerpOp(float w) : w(w) {}

  __device__ __forceinline__ void operator()(float *out, float *a, float *b) {
    *out = *a + w * (*b - *a);
  }

  const float w;
};

void THCudaTensor_lerp(THCState *state, THCudaTensor *result, THCudaTensor *a, THCudaTensor *b, float w)
{
  THAssert(THCudaTensor_checkGPU(state, 3, result, a, b));
  THArgCheck(THCudaTensor_nElement(state, a) ==
             THCudaTensor_nElement(state, b), 3, "sizes do not match");
  THCudaTensor_resizeAs(state, result, a);

  if (!THC_pointwiseApply3(state, result, a, b, TensorLerpOp(w))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

struct dist_functor
{
  const float exponent;

  dist_functor(float exponent_) : exponent(exponent_) {}

  __host__ __device__ float operator()(const float& x, const float& y) const
  {
    return pow(fabs(x-y), exponent);
  }
};

float THCudaTensor_dist(THCState *state, THCudaTensor *self, THCudaTensor *src, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self, src));
  self = THCudaTensor_newContiguous(state, self);
  long size = THCudaTensor_nElement(state, self);
  src = THCudaTensor_newContiguous(state, src);
  thrust::device_ptr<float> self_data(THCudaTensor_data(state, self));
  thrust::device_ptr<float> src_data(THCudaTensor_data(state, src));

  float result = thrust::inner_product(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    self_data, self_data+size, src_data, (float) 0,
    thrust::plus<float>(), dist_functor(value));

  THCudaTensor_free(state, src);
  THCudaTensor_free(state, self);

  return pow(result, (float)1.0/value);
}

void THCudaTensor_rand(THCState *state, THCudaTensor *r_, THLongStorage *size)
{
  THAssert(THCudaTensor_checkGPU(state, 1, r_));
  THCudaTensor_resize(state, r_, size, NULL);
  THCudaTensor_uniform(state, r_, 0, 1);
}

void THCudaTensor_randn(THCState *state, THCudaTensor *r_, THLongStorage *size)
{
  THAssert(THCudaTensor_checkGPU(state, 1, r_));
  THCudaTensor_resize(state, r_, size, NULL);
  THCudaTensor_normal(state, r_, 0, 1);
}

struct TensorCrossOp {
  TensorCrossOp(long sx, long sy, long so) : sx(sx), sy(sy), so(so) {}

  __device__ __forceinline__ void operator()(float* out, float* x, float*y) {
    out[0 * so] = x[1 * sx] * y[2 * sy] - x[2 * sx] * y[1 * sy];
    out[1 * so] = x[2 * sx] * y[0 * sy] - x[0 * sx] * y[2 * sy];
    out[2 * so] = x[0 * sx] * y[1 * sy] - x[1 * sx] * y[0 * sy];
  }

  const long sx, sy, so;
};

THC_API void THCudaTensor_cross(THCState *state, THCudaTensor *self, THCudaTensor *x, THCudaTensor *y, int dimension)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self, x, y));

  int i;
  long nd = THCudaTensor_nDimension(state, x);
  long nelem = THCudaTensor_nElement(state, x);
  THArgCheck(nd == THCudaTensor_nDimension(state, y), 1, "tensors must have same number of dimensions");
  for (i = 0; i < nd; i++) {
    THArgCheck(THCudaTensor_size(state, x, i) == THCudaTensor_size(state, y, i), 1, "dimension %i of x and y does not match", i);
    if (dimension < 0 && THCudaTensor_size(state, x, i) == 3) {
      dimension = i;
    }
  }

  THArgCheck(dimension >= 0 && dimension < nd, 3, "dimension %d out of range", dimension+1);
  THArgCheck(THCudaTensor_size(state, x, dimension) == 3, 3,
      "dimension %d does not have size 3", dimension+1);
  THCudaTensor_resizeAs(state, self, x);

  long sx = THCudaTensor_stride(state, x, dimension);
  long sy = THCudaTensor_stride(state, y, dimension);
  long so = THCudaTensor_stride(state, self, dimension);
  THCudaTensor *nx = THCudaTensor_newNarrow(state, x, dimension, 0, 1);
  THCudaTensor *ny = THCudaTensor_newNarrow(state, y, dimension, 0, 1);
  THCudaTensor *nself = THCudaTensor_newNarrow(state, self, dimension, 0, 1);
  if (!THC_pointwiseApply3(state, nself, nx, ny, TensorCrossOp(sx, sy, so))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }
  THCudaTensor_free(state, nx);
  THCudaTensor_free(state, ny);
  THCudaTensor_free(state, nself);
}
