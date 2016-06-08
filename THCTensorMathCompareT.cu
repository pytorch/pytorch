#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"

template<class Op>
void THCudaTensor_logicalTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2, Op op)
{
  THCudaTensor_resizeAs(state, self_, src1);
  THArgCheck(THCudaTensor_nElement(state, src1) == THCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (!THC_pointwiseApply3(state, self_, src1, src2, op)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorLTOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a < *b);
  }
};

struct TensorGTOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a > *b);
  }
};

struct TensorLEOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a <= *b);
  }
};

struct TensorGEOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a >= *b);
  }
};

struct TensorEQOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a == *b);
  }
};

struct TensorNEOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a != *b);
  }
};

void THCudaTensor_ltTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THCudaTensor_logicalTensor(state, self_, src1, src2, TensorLTOp());
}


void THCudaTensor_gtTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THCudaTensor_logicalTensor(state, self_, src1, src2, TensorGTOp());
}


void THCudaTensor_leTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THCudaTensor_logicalTensor(state, self_, src1, src2, TensorLEOp());
}


void THCudaTensor_geTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THCudaTensor_logicalTensor(state, self_, src1, src2, TensorGEOp());
}


void THCudaTensor_eqTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THCudaTensor_logicalTensor(state, self_, src1, src2, TensorEQOp());
}


void THCudaTensor_neTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THCudaTensor_logicalTensor(state, self_, src1, src2, TensorNEOp());
}
