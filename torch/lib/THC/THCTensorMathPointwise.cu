#include "THCTensorMathPointwise.cuh"

struct TensorMaxOp {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = max(*out, *in);
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = max(*in1, *in2);
  }
};

void THCudaTensor_cmax(THCState *state, THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self, src1, src2));
  THArgCheck(THCudaTensor_nElement(state, src1) ==
             THCudaTensor_nElement(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2(state, self, src2, TensorMaxOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self, src1);
    if (!THC_pointwiseApply3(state, self, src1, src2, TensorMaxOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

struct TensorMinOp {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = min(*out, *in);
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = min(*in1, *in2);
  }
};

void THCudaTensor_cmin(THCState *state, THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self, src1, src2));
  THArgCheck(THCudaTensor_nElement(state, src1) ==
             THCudaTensor_nElement(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2(state, self, src2, TensorMinOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self, src1);
    if (!THC_pointwiseApply3(state, self, src1, src2, TensorMinOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

struct TensorMaxValueOp {
  TensorMaxValueOp(float v) : val(v) {}

  __device__ __forceinline__ void operator()(float* out) {
    *out = max(*out, val);
  }

  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = max(*in, val);
  }

  float val;
};

void THCudaTensor_cmaxValue(THCState *state, THCudaTensor *self, THCudaTensor *src, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self, src));

  if (self == src) {
    if (!THC_pointwiseApply1(state, self, TensorMaxValueOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self, src);
    if (!THC_pointwiseApply2(state, self, src, TensorMaxValueOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

struct TensorMinValueOp {
  TensorMinValueOp(float v) : val(v) {}

  __device__ __forceinline__ void operator()(float* out) {
    *out = min(*out, val);
  }

  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = min(*in, val);
  }

  float val;
};

void THCudaTensor_cminValue(THCState *state, THCudaTensor *self, THCudaTensor *src, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self, src));

  if (self == src) {
    if (!THC_pointwiseApply1(state, self, TensorMinValueOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self, src);
    if (!THC_pointwiseApply2(state, self, src, TensorMinValueOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}
