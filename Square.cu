#include "utils.h"
#include "THCApply.cuh"

struct squareupdateOutput_functor
{
  __device__ void operator()(float* output, const float* input) const
  {
    *output = *input* *input;
  }
};

static int cunn_Square_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 2, input, output));
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input, squareupdateOutput_functor());
  return 1;
}

struct squareupdateGradInput_functor
{
  __device__ void operator()(float* gradInput, const float* input, const float* gradOutput) const
  {
    *gradInput = 2.0 * *gradOutput * *input;
  }
};

static int cunn_Square_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 3, input, gradOutput, gradInput));
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput, squareupdateGradInput_functor());
  return 1;
}

static const struct luaL_Reg cunn_Square__ [] = {
  {"Square_updateOutput", cunn_Square_updateOutput},
  {"Square_updateGradInput", cunn_Square_updateGradInput},
  {NULL, NULL}
};

void cunn_Square_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Square__, "nn");
  lua_pop(L,1);
}
