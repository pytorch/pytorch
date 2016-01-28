#include "utils.h"
#include "THCApply.cuh"

struct SoftShrinkUpdateOutput {
  const float lambda_;

  SoftShrinkUpdateOutput(float lambda): lambda_(lambda){}

  __device__ __forceinline__ void operator()(float* out, float* in) {
    float x = *in;
    if (x > lambda_) *out = x - lambda_;
    else if (x < -lambda_) *out = x + lambda_;
    else *out = 0;
  }
};

static int cunn_SoftShrink_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  double lambda = luaT_getfieldchecknumber(L, 1, "lambda");
  THAssert(THCudaTensor_checkGPU(state, 2, input, output));
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input, SoftShrinkUpdateOutput(lambda));
  THCudaCheck(cudaGetLastError());
  return 1;
}

struct SoftShrinkUpdateGradInput
{
  const float lambda_;

  SoftShrinkUpdateGradInput(float lambda) : lambda_(lambda) {}

  __device__ __forceinline__ void operator()(float* gradInput, float* input,
      float* gradOutput) const {
    float x = *input;
    if (x > lambda_ || x < -lambda_)
      *gradInput = *gradOutput;
    else
      *gradInput = 0;
  }
};


static int cunn_SoftShrink_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  double lambda = luaT_getfieldchecknumber(L, 1, "lambda");
  THAssert(THCudaTensor_checkGPU(state, 3, input, gradOutput, gradInput));
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput, SoftShrinkUpdateGradInput(lambda));
  THCudaCheck(cudaGetLastError());
  return 1;
}

static const struct luaL_Reg cunn_SoftShrink__ [] = {
  {"SoftShrink_updateOutput", cunn_SoftShrink_updateOutput},
  {"SoftShrink_updateGradInput", cunn_SoftShrink_updateGradInput},
  {NULL, NULL}
};

void cunn_SoftShrink_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SoftShrink__, "nn");
  lua_pop(L,1);
}
