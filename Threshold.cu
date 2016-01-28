#include "THCApply.cuh"
#include "utils.h"

struct ThresholdUpdateOutput {
  const float threshold_;
  const float val_;

  ThresholdUpdateOutput(float threshold, float val): threshold_(threshold),
                                                     val_(val) {}

  __device__ __forceinline__ void operator()(float* out, float* in) {
    float x = *in;
    *out = (x > threshold_) ? x : val_;
  }
};

// in-place variant
struct ThresholdUpdateOutputIP {
  const float threshold_;
  const float val_;

  ThresholdUpdateOutputIP(float threshold, float val): threshold_(threshold),
                                                       val_(val) {}

  __device__ __forceinline__ void operator()(float* x) {
    *x = (*x > threshold_) ? *x : val_;
  }
};

static int cunn_Threshold_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  double val = luaT_getfieldchecknumber(L, 1, "val");
  double threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  bool   inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");

  THAssert(THCudaTensor_checkGPU(state, 2, input, output));

  if (inPlace) {
    THCudaTensor_pointwiseApply1(state, input,
                                 ThresholdUpdateOutputIP(threshold, val));
    THCudaTensor_set(state, output, input);
  } else {
    THCudaTensor_resizeAs(state, output, input);
    THCudaTensor_pointwiseApply2(state, output, input,
                                 ThresholdUpdateOutput(threshold, val));
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}

struct ThresholdUpdateGradInput
{
  const float threshold_;

  ThresholdUpdateGradInput(float threshold) : threshold_(threshold) {}

  __device__ __forceinline__ void operator()(float* gradInput, float* input,
                                             float* gradOutput) const {
    *gradInput = (*input > threshold_) ? *gradOutput : 0;
  }
};

struct ThresholdUpdateGradInputIP
{
  const float threshold_;

  ThresholdUpdateGradInputIP(float threshold) : threshold_(threshold) {}

  __device__ __forceinline__ void operator()(float* gradOutput,
                                             float* input) const {
    *gradOutput = (*input > threshold_) ? *gradOutput : 0;
  }
};

static int cunn_Threshold_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  double val = luaT_getfieldchecknumber(L, 1, "val");
  double threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  bool   inPlace   = luaT_getfieldcheckboolean(L, 1, "inplace");

  THAssert(THCudaTensor_checkGPU(state, 4, input, output, gradInput, gradOutput));

  if (inPlace) {
    THCudaTensor_pointwiseApply2(state, gradOutput, input,
                                 ThresholdUpdateGradInputIP(threshold));
    THCudaTensor_set(state, gradInput, gradOutput);
  } else {
    THCudaTensor_resizeAs(state, gradInput, output);
    THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput,
                                 ThresholdUpdateGradInput(threshold));
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}

static const struct luaL_Reg cunn_Threshold__ [] = {
  {"Threshold_updateOutput", cunn_Threshold_updateOutput},
  {"Threshold_updateGradInput", cunn_Threshold_updateGradInput},
  {NULL, NULL}
};

void cunn_Threshold_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Threshold__, "nn");
  lua_pop(L,1);
}
