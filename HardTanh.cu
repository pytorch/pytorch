#include "utils.h"
#include "THCApply.cuh"

struct hardtanhupdateOutput_functor
{
  const float max_val_;
  const float min_val_;

  hardtanhupdateOutput_functor(float min_val, float max_val): min_val_(min_val),
                                                              max_val_(max_val) {}

  __device__ void operator()(float* output, const float* input) const
  {
    if(*input < min_val_)
      *output = min_val_;
    else if(*input <= max_val_)
      *output = *input;
    else
      *output = max_val_;
  }
};

static int cunn_HardTanh_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  float min_val = luaT_getfieldchecknumber(L, 1, "min_val");
  float max_val = luaT_getfieldchecknumber(L, 1, "max_val");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 2, input, output));
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input,
                               hardtanhupdateOutput_functor(min_val, max_val));
  return 1;
}

struct hardtanhupdateGradInput_functor
{
  const float max_val_;
  const float min_val_;

  hardtanhupdateGradInput_functor(float min_val, float max_val): min_val_(min_val),
                                                                 max_val_(max_val) {}

  __device__ void operator()(float* gradInput, const float* input, const float* gradOutput) const
  {
    if(*input < min_val_ || *input > max_val_)
      *gradInput = 0;
    else
      *gradInput = *gradOutput;
  }
};

static int cunn_HardTanh_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  float min_val = luaT_getfieldchecknumber(L, 1, "min_val");
  float max_val = luaT_getfieldchecknumber(L, 1, "max_val");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 3, input, gradOutput, gradInput));

  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput,
                               hardtanhupdateGradInput_functor(min_val, max_val));
  return 1;
}

static const struct luaL_Reg cunn_HardTanh__ [] = {
  {"HardTanh_updateOutput", cunn_HardTanh_updateOutput},
  {"HardTanh_updateGradInput", cunn_HardTanh_updateGradInput},
  {NULL, NULL}
};

void cunn_HardTanh_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_HardTanh__, "nn");
  lua_pop(L,1);
}
