#include "utils.h"

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

struct kl_functor
{
  kl_functor() {}

  __host__ __device__ float operator()(const float& x, const float& y) const
    {
      return y > 0 ? y * (log(y) - x) : 0;
  }
};


static int cunn_DistKLDivCriterion_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 2, input, target));

  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  luaL_argcheck(L, THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2,
                "input and target need to have the same number of elements");

  float sum;

  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  sum = thrust::inner_product(input_data, input_data+size, target_data, (float) 0, thrust::plus<float>(), kl_functor());

  if(sizeAverage)
    sum /= size;

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}


struct kl_updateGradInput_functor
{
  const float norm;

  kl_updateGradInput_functor(float norm_) : norm(norm_) {}

  __host__ __device__ float operator()(const float& x, const float& y) const
    {
      return y > 0 ? norm * (-y) : 0;
  }
};

static int cunn_DistKLDivCriterion_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 3, input, target, gradInput));

  luaL_argcheck(L, THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2,
                "input and target need to have the same number of elements");

  long size = THCudaTensor_nElement(state, input);
  float norm = (sizeAverage ? 2./size : 2.);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  THCudaTensor_resizeAs(state, gradInput, input);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));

  thrust::transform(input_data, input_data+size, target_data, gradInput_data, kl_updateGradInput_functor(norm));

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
  return 1;
}

static const struct luaL_Reg cunn_DistKLDivCriterion__ [] = {
  {"DistKLDivCriterion_updateOutput", cunn_DistKLDivCriterion_updateOutput},
  {"DistKLDivCriterion_updateGradInput", cunn_DistKLDivCriterion_updateGradInput},
  {NULL, NULL}
};

void cunn_DistKLDivCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_DistKLDivCriterion__, "nn");
  lua_pop(L,1);
}
