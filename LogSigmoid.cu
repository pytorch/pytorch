#include "utils.h"
#include "THCApply.cuh"

struct logSigmoid_updateOutput_functor
{
  __device__ void operator()(float* output, const float* input) const
  {
    float z = exp(-*input);
    *output = -log(1. + z);
  }
};

static int cunn_LogSigmoid_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 2, input, output));
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input, logSigmoid_updateOutput_functor());
  return 1;
}

struct logSigmoid_updateGradInput_functor
{
  __device__ void operator()(float* gradInput, const float* input, const float* gradOutput) const
  {
    float z = exp(-*input);
    *gradInput = *gradOutput * z / (1. + z);
  }
};

static int cunn_LogSigmoid_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 3, input, gradOutput, gradInput));
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput, logSigmoid_updateGradInput_functor());
  return 1;
}

static const struct luaL_Reg cunn_LogSigmoid__ [] = {
  {"LogSigmoid_updateOutput", cunn_LogSigmoid_updateOutput},
  {"LogSigmoid_updateGradInput", cunn_LogSigmoid_updateGradInput},
  {NULL, NULL}
};

void cunn_LogSigmoid_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_LogSigmoid__, "nn");
  lua_pop(L,1);
}
