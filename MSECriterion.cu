#include "utils.h"

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

struct mse_functor
{
  mse_functor() {}

  __host__ __device__ float operator()(const float& x, const float& y) const
    {
      float z = x-y;
      return z*z;
  }
};


static int cunn_MSECriterion_updateOutput(lua_State *L)
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
  sum = thrust::inner_product(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, (float) 0,
    thrust::plus<float>(), mse_functor());

  if(sizeAverage)
    sum /= size;

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}


struct mse_updateGradInput_functor
{
  const float norm;

  mse_updateGradInput_functor(float norm_) : norm(norm_) {}

  __host__ __device__ float operator()(const float& x, const float& y) const
    {
      return norm * (x - y);
  }
};

static int cunn_MSECriterion_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  luaL_argcheck(L, THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2,
                "input and target need to have the same number of elements");
  THAssert(THCudaTensor_checkGPU(state, 3, input, target, gradInput));

  long size = THCudaTensor_nElement(state, input);
  float norm = (sizeAverage ? 2./size : 2.);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  THCudaTensor_resizeAs(state, gradInput, input);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));

  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, gradInput_data,
    mse_updateGradInput_functor(norm));

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
  return 1;
}

#define MSECRITERION_THREADS 128

__global__ void cunn_MSECriterion_updateOutput_kernel(float* output, float *input, float *target, int nframe, int dim, int sizeAverage)
{
  __shared__ float buffer[MSECRITERION_THREADS];
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *target_k = target + k*dim;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // mse
  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step)
  {
    float z = input_k[i] - target_k[i];
    buffer[threadIdx.x] += z*z;
  }
  __syncthreads();


  //reduce
  if (threadIdx.x == 0)
  {
    *output = 0;
    for (int i=0; i<blockDim.x; i++)
    {
      *output += buffer[i];
    }
    if (sizeAverage)
      *output /= dim;
  }
}


__global__ void cunn_MSECriterion_updateGradInput_kernel(float *gradInput, float *input, float *target, float norm, int nframe, int dim)
{
  int k = blockIdx.x;
  float *gradInput_k = gradInput + k*dim;
  float *input_k = input + k*dim;
  float *target_k = target + k*dim;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // gradInput
  for (int i=i_start; i<i_end; i+=i_step)
    gradInput_k[i] = norm*(input_k[i] - target_k[i]);
}

static int cunn_MSECriterion_updateOutput2(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  THCudaStorage *output = THCudaStorage_newWithSize(state, 1);

  dim3 blocks(1);
  dim3 threads(MSECRITERION_THREADS);

  cunn_MSECriterion_updateOutput_kernel<<<blocks,threads,
    0, THCState_getCurrentStream(state)>>>(output->data,
                                           THCudaTensor_data(state, input),
                                           THCudaTensor_data(state, target),
                                           1, size,
                                           sizeAverage);

  lua_pushnumber(L, THCudaStorage_get(state, output, 0));

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
  THCudaStorage_free(state, output);

  lua_pushstring(L, "output");
  lua_pushvalue(L, -2);
  lua_rawset(L, 1);

  return 1;
}

static int cunn_MSECriterion_updateGradInput2(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long size = THCudaTensor_nElement(state, input);
  float norm = (sizeAverage ? 2./size : 2.);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  THCudaTensor_resizeAs(state, gradInput, input);

  dim3 blocks(1);
  dim3 threads(MSECRITERION_THREADS);

  cunn_MSECriterion_updateGradInput_kernel<<<blocks,threads,
    0, THCState_getCurrentStream(state)>>>(THCudaTensor_data(state, gradInput),
                                           THCudaTensor_data(state, input),
                                           THCudaTensor_data(state, target),
                                           norm,
                                           1, size);

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
  return 1;
}


static const struct luaL_Reg cunn_MSECriterion__ [] = {
  {"MSECriterion_updateOutput", cunn_MSECriterion_updateOutput},
  {"MSECriterion_updateGradInput", cunn_MSECriterion_updateGradInput},
  {"MSECriterion_updateOutput2", cunn_MSECriterion_updateOutput2},
  {"MSECriterion_updateGradInput2", cunn_MSECriterion_updateGradInput2},
  {NULL, NULL}
};

void cunn_MSECriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_MSECriterion__, "nn");
  lua_pop(L,1);
}
