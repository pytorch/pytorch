#include "utils.h"
#include "THCApply.cuh"

struct softPlusupdateOutput_functor
{
  const float threshold;
  const float beta;

  softPlusupdateOutput_functor(float threshold_, float beta_) : threshold(threshold_), beta(beta_) {}

  __device__ void operator()(float* output, const float* input) const
  {
    float betain = beta * *input;
    *output = ((betain) > threshold) ? *input : (1/beta) * log1p(exp(betain));
  }
};

static int cunn_SoftPlus_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  float beta = luaT_getfieldchecknumber(L, 1, "beta");
  float threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  THAssert(THCudaTensor_checkGPU(state, 2, input, output));
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input, softPlusupdateOutput_functor(threshold, beta));
  return 1;
}

struct softPlusupdateGradInput_functor
{
  const float threshold;
  const float beta;

  softPlusupdateGradInput_functor(float threshold_, float beta_) : threshold(threshold_), beta(beta_) {}

  __device__ void operator()(float* gradInput, const float* output, const float* gradOutput) const
  {
    float betaout = beta * *output;
    float exp_bo = exp(betaout);
    *gradInput = ((betaout) > threshold) ? *gradOutput : *gradOutput * (exp_bo - 1) / exp_bo;
  }
};

static int cunn_SoftPlus_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 4, input, output, gradOutput, gradInput));
  float beta = luaT_getfieldchecknumber(L, 1, "beta");
  float threshold = luaT_getfieldchecknumber(L, 1, "threshold");

  THCudaTensor_resizeAs(state, gradInput, output);
  THCudaTensor_pointwiseApply3(state, gradInput, output, gradOutput, softPlusupdateGradInput_functor(threshold, beta));
  return 1;
}

static const struct luaL_Reg cunn_SoftPlus__ [] = {
  {"SoftPlus_updateOutput", cunn_SoftPlus_updateOutput},
  {"SoftPlus_updateGradInput", cunn_SoftPlus_updateGradInput},
  {NULL, NULL}
};

void cunn_SoftPlus_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SoftPlus__, "nn");
  lua_pop(L,1);
}
