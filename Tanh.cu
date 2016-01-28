#include "utils.h"
#include "THCApply.cuh"

struct tanhupdateOutput_functor
{
  __device__ void operator()(float* output, const float* input) const
  {
    *output = tanh(*input);
  }
};

static int cunn_Tanh_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 2, input, output));
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input, tanhupdateOutput_functor());
  return 1;
}

struct tanhupdateGradInput_functor
{
  __device__ void operator()(float* gradInput, const float* output, const float* gradOutput) const
  {
    *gradInput = *gradOutput * (1 - *output * *output);
  }
};

static int cunn_Tanh_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 3, output, gradOutput, gradInput));
  THCudaTensor_resizeAs(state, gradInput, output);
  THCudaTensor_pointwiseApply3(state, gradInput, output, gradOutput, tanhupdateGradInput_functor());
  return 1;
}

static const struct luaL_Reg cunn_Tanh__ [] = {
  {"Tanh_updateOutput", cunn_Tanh_updateOutput},
  {"Tanh_updateGradInput", cunn_Tanh_updateGradInput},
  {NULL, NULL}
};

void cunn_Tanh_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Tanh__, "nn");
  lua_pop(L,1);
}
