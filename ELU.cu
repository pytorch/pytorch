#include "utils.h"
#include "THCApply.cuh"

struct ELUupdateOutput_functor
{
  const float alpha_;

  ELUupdateOutput_functor(float alpha) : alpha_(alpha) {}

  __device__ void operator()(float* output, const float* input) const
  {
    *output = *input <= 0 ? (exp(*input)-1)*alpha_ : *input;
  }
};

static int cunn_ELU_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  float alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 2, input, output));
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input, ELUupdateOutput_functor(alpha));
  return 1;
}

struct ELUupdateGradInput_functor
{
  const float alpha_;

  ELUupdateGradInput_functor(float alpha) : alpha_(alpha) {}

  __device__ void operator()(float* gradInput, const float* output, const float* gradOutput) const
  {
    *gradInput = (*output) <= 0 ? (*gradOutput * (*output + alpha_)) : (*gradOutput);
  }
};

static int cunn_ELU_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  float alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 3, output, gradOutput, gradInput));
  THCudaTensor_resizeAs(state, gradInput, output);
  THCudaTensor_pointwiseApply3(state, gradInput, output, gradOutput, ELUupdateGradInput_functor(alpha));
  return 1;
}

static const struct luaL_Reg cunn_ELU__ [] = {
  {"ELU_updateOutput", cunn_ELU_updateOutput},
  {"ELU_updateGradInput", cunn_ELU_updateGradInput},
  {NULL, NULL}
};

void cunn_ELU_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_ELU__, "nn");
  lua_pop(L,1);
}
