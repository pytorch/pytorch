#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"

#include <cfloat>

void THCudaTensor_cat(THCState *state, THCudaTensor *result, THCudaTensor *ta, THCudaTensor *tb, int dimension)
{
  THCudaTensor* inputs[2];
  inputs[0] = ta;
  inputs[1] = tb;
  THCudaTensor_catArray(state, result, inputs, 2, dimension);
}

void THCudaTensor_catArray(THCState *state, THCudaTensor *result, THCudaTensor **inputs, int numInputs, int dimension)
{
  THLongStorage *size;
  int i, j;
  long offset;
  int ndim = dimension + 1;
  for (i = 0; i < numInputs; i++)
  {
    ndim = THMax(ndim, THCudaTensor_nDimension(state, inputs[i]));
  }

  THArgCheck(numInputs > 0, 3, "invalid number of inputs %d", numInputs);
  THArgCheck(dimension >= 0, 4, "invalid dimension %d", dimension+1);

  size = THLongStorage_newWithSize(ndim);
  for(i = 0; i < ndim; i++)
  {
    long dimSize = i < THCudaTensor_nDimension(state, inputs[0])
                       ? THCudaTensor_size(state, inputs[0], i)
                       : 1;
    if (i == dimension)
    {
      for (j = 1; j < numInputs; j++)
      {
        dimSize += i < THCudaTensor_nDimension(state, inputs[j])
                       ? THCudaTensor_size(state, inputs[j], i)
                       : 1;
      }
    }
    else
    {
      for (j = 1; j < numInputs; j++)
      {
        if (dimSize != (i < THCudaTensor_nDimension(state, inputs[j])
                            ? THCudaTensor_size(state, inputs[j], i)
                            : 1)) {
          THLongStorage_free(size);
          THError("inconsistent tensor sizes");
        }
      }
    }
    size->data[i] = dimSize;
  }

  THCudaTensor_resize(state, result, size, NULL);
  THLongStorage_free(size);

  offset = 0;
  for (j = 0; j < numInputs; j++)
  {
    long dimSize = dimension < THCudaTensor_nDimension(state, inputs[j])
                       ? THCudaTensor_size(state, inputs[j], dimension)
                       : 1;
    THCudaTensor *nt = THCudaTensor_newWithTensor(state, result);
    THCudaTensor_narrow(state, nt, NULL, dimension, offset, dimSize);
    THCudaTensor_copy(state, nt, inputs[j]);
    THCudaTensor_free(state, nt);
    offset += dimSize;
  }
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
  else
  {
    THArgCheck(THCudaTensor_nElement(state, self_) == THCudaTensor_nElement(state, src1),
               1, "sizes do not match");
  }

  THArgCheck(THCudaTensor_nElement(state, src1) == THCudaTensor_nElement(state, src2),
             3, "sizes do not match");

  if (!THC_pointwiseApply3(state, self_, src1, src2, TensorAddCMulOp(value))) {
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
  else
  {
    THArgCheck(THCudaTensor_nElement(state, self_) == THCudaTensor_nElement(state, src1),
               1, "sizes do not match");
  }
  THArgCheck(THCudaTensor_nElement(state, src1) == THCudaTensor_nElement(state, src2),
             3, "sizes do not match");

  if (!THC_pointwiseApply3(state, self_, src1, src2, TensorAddCDivOp(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

template <typename T>
struct TensorFillOp {
  TensorFillOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* v) { *v = val; }

  const T val;
};

#include "generic/THCTensorMath.cu"
#include "THCGenerateAllTypes.h"
