#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"

template<class Op>
void THCudaTensor_logicalValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, Op op)
{
  THCudaTensor_resizeAs(state, self_, src);

  if (!THC_pointwiseApply2(state, self_, src, op)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorLTValueOp {
  TensorLTValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in < value);
  }

  const float value;
};

void THCudaTensor_ltValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src));
  THCudaTensor_logicalValue(state, self_, src, TensorLTValueOp(value));
}

struct TensorGTValueOp {
  TensorGTValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in > value);
  }

  const float value;
};

void THCudaTensor_gtValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src));
  THCudaTensor_logicalValue(state, self_, src, TensorGTValueOp(value));
}

struct TensorLEValueOp {
  TensorLEValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in <= value);
  }

  const float value;
};

void THCudaTensor_leValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src));
  THCudaTensor_logicalValue(state, self_, src, TensorLEValueOp(value));
}

struct TensorGEValueOp {
  TensorGEValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in >= value);
  }

  const float value;
};

void THCudaTensor_geValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src));
  THCudaTensor_logicalValue(state, self_, src, TensorGEValueOp(value));
}

struct TensorEQValueOp {
  TensorEQValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in == value);
  }

  const float value;
};

void THCudaTensor_eqValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src));
  THCudaTensor_logicalValue(state, self_, src, TensorEQValueOp(value));
}

struct TensorNEValueOp {
  TensorNEValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in != value);
  }

  const float value;
};

void THCudaTensor_neValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src));
  THCudaTensor_logicalValue(state, self_, src, TensorNEValueOp(value));
}
