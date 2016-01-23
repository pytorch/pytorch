#include "THCApply.cuh"
#include "THCReduce.cuh"
#include "utils.h"
#include "common.h"

#include <thrust/functional.h>

struct PReLUUpdateOutput {
  float* weight_;

  PReLUUpdateOutput(float* weight): weight_(weight) {}

  __device__ __forceinline__ void operator()(float* out, float* in) {
    float x = *in;
    *out = (x > 0) ? x : weight_[0] * x;
  }
};

__global__ void preluForward(float *output, const float *input, const float *weight,
    int n, int nElemsPerSample, int mapSize)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    int positionInSample = i % nElemsPerSample;
    int mapNumber = positionInSample / mapSize;
    output[i] = input[i] > 0 ? input[i] : input[i] * weight[mapNumber];
  }
}


static int cunn_PReLU_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  // Input:
  THCudaTensor *input = (THCudaTensor*) luaT_checkudata(L, 2, "torch.CudaTensor");
  // Params:
  THCudaTensor *weight = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  long nOutputPlane = luaT_getfieldchecknumber(L, 1, "nOutputPlane");

  THCudaTensor_resizeAs(state, output, input);

  float* w = THCudaTensor_data(state, weight);

  if(nOutputPlane == 0)
    THCudaTensor_pointwiseApply2(state, output, input, PReLUUpdateOutput(w));
  else
  {
    int ndim = THCudaTensor_nDimension(state, input);
    input = THCudaTensor_newContiguous(state, input);

    int n = THCudaTensor_nElement(state, input);
    int mapSize = 1;
    if(ndim == 3)
      mapSize = (input->size[1] * input->size[2]);
    else if(ndim == 4)
      mapSize = (input->size[2] * input->size[3]);
    int nElemsPerSample = nOutputPlane * mapSize;
    preluForward<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
	THCudaTensor_data(state, output),
	THCudaTensor_data(state, input),
	w,
	n, nElemsPerSample, mapSize);
    THCudaTensor_free(state, input);
  }

  return 1;
}

struct PReLUUpdateGradInput {
  float *weight_;

  PReLUUpdateGradInput(float* weight): weight_(weight) {}

  __device__ __forceinline__ void operator()(float* gradInput,
                                             float* gradOutput,
                                             float* input) {
    *gradInput = *input > 0 ? *gradOutput : *gradOutput * *weight_;
  }
};

__global__ void preluBackward(
    float *gradInput,
    const float *input,
    const float *weight,
    const float *gradOutput,
    int n, int nElemsPerSample, int mapSize)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    int positionInSample = i % nElemsPerSample;
    int mapNumber = positionInSample / mapSize;
    gradInput[i] = input[i] > 0 ? gradOutput[i] : gradOutput[i] * weight[mapNumber];
  }
}

static int cunn_PReLU_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  // Input:
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  // Params:
  THCudaTensor *weight = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long nOutputPlane = luaT_getfieldchecknumber(L, 1, "nOutputPlane");

  THCudaTensor_resizeAs(state, gradInput, input);

  float* w = THCudaTensor_data(state, weight);
  if(nOutputPlane == 0)
    THCudaTensor_pointwiseApply3(state, gradInput, gradOutput, input, PReLUUpdateGradInput(w));
  else
  {
    int ndim = THCudaTensor_nDimension(state, input);
    input = THCudaTensor_newContiguous(state, input);
    gradOutput = THCudaTensor_newContiguous(state, gradOutput);

    int n = THCudaTensor_nElement(state, input);
    int mapSize = 1;
    if(ndim == 3)
      mapSize = (input->size[1] * input->size[2]);
    else if(ndim == 4)
      mapSize = (input->size[2] * input->size[3]);
    int nElemsPerSample = nOutputPlane * mapSize;
    preluBackward<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
	THCudaTensor_data(state, gradInput),
	THCudaTensor_data(state, input),
	w,
	THCudaTensor_data(state, gradOutput),
	n, nElemsPerSample, mapSize);

    THCudaTensor_free(state, input);
    THCudaTensor_free(state, gradOutput);
  }
  return 1;
}

struct PReLUAccGradParametersShared {
  __device__ __forceinline__ void operator()(float* gradInput,
      					     float* input,
                                             float* gradOutput) {
    *gradInput = (*input) * (*gradOutput) * (*input <= 0);
  }
};

struct PReLUAccGradParameters {
  float scale;
  PReLUAccGradParameters(float scale) : scale(scale) {}

  __device__ __forceinline__ void operator()(float* gradInput,
      					     float* input,
                                             float* gradOutput) {
    *gradInput = (*input) * (*gradOutput) * scale * (*input <= 0);
  }
};

struct PReLUAccGradParameters1to1 {
  float scale;
  PReLUAccGradParameters1to1(float scale) : scale(scale) {}

  __device__ __forceinline__ void operator()(float* gradWeight,
      					     float* input,
                                             float* gradOutput) {
    *gradWeight += (*input) * (*gradOutput) * scale * (*input <= 0);
  }
};

static int cunn_PReLU_accGradParameters(lua_State *L)
{
  THCState *state = getCutorchState(L);
  // Input:
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  // Params:
  THCudaTensor *weight = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradWeight = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long nOutputPlane = luaT_getfieldchecknumber(L, 1, "nOutputPlane");
  float scale = luaL_optnumber(L, 4, 1);

  // use grad input for temporary storage, then call updateGradInput again

  if(nOutputPlane == 0)
  {
    THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput, PReLUAccGradParametersShared());
    // introduces a sync point
    float sum = THCudaTensor_sumall(state, gradInput);
    float weight = THCudaTensor_get1d(state, gradWeight, 0);
    THCudaTensor_set1d(state, gradWeight, 0, weight + sum * scale);

    // restore gradInput
    cunn_PReLU_updateGradInput(L);
  }
  else
  {
    int ndim = THCudaTensor_nDimension(state, input);

    if(ndim == 1)
      THCudaTensor_pointwiseApply3(state, gradWeight, input, gradOutput, PReLUAccGradParameters1to1(scale));
    else
    {
      THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput, PReLUAccGradParameters(scale));

      THCudaTensor *gradWeightBuf = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "gradWeightBuf", "torch.CudaTensor");
      THCudaTensor *sumbuf = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "gradWeightBuf2", "torch.CudaTensor");
      THCudaTensor_resizeAs(state, gradWeightBuf, gradWeight);

      if(ndim == 2)
      {
	THCudaTensor_sum(state, gradWeightBuf, gradInput, 0);
	THCudaTensor_cadd(state, gradWeight, gradWeight, scale, gradWeightBuf);
      }
      else if(ndim == 3)
      {
	THCudaTensor *buffer = THCudaTensor_newContiguous(state, gradInput);
	THCudaTensor_resize2d(state, buffer, nOutputPlane, input->size[1] * input->size[2]);
	THCudaTensor_sum(state, gradWeightBuf, buffer, 1);
	THCudaTensor_cadd(state, gradWeight, gradWeight, scale, gradWeightBuf);
	THCudaTensor_free(state, buffer);
      }
      else if(ndim == 4)
      {
	THCudaTensor *buffer = THCudaTensor_newContiguous(state, gradInput);
	THCudaTensor_resize3d(state, buffer, input->size[0], nOutputPlane, input->size[2] * input->size[3]);
	THCudaTensor_resize2d(state, sumbuf, input->size[0], nOutputPlane);
	THCudaTensor_sum(state, sumbuf, buffer, 2);
	THCudaTensor_sum(state, gradWeightBuf, sumbuf, 0);
	THCudaTensor_cadd(state, gradWeight, gradWeight, scale, gradWeightBuf);
	THCudaTensor_free(state, buffer);
      }

      // restore gradInput
      cunn_PReLU_updateGradInput(L);
    }
  }

  return 1;
}

static const struct luaL_Reg cunn_PRelu__ [] = {
  {"PReLU_updateOutput", cunn_PReLU_updateOutput},
  {"PReLU_updateGradInput", cunn_PReLU_updateGradInput},
  {"PReLU_accGradParameters", cunn_PReLU_accGradParameters},
  {NULL, NULL}
};

void cunn_PReLU_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_PRelu__, "nn");
  lua_pop(L,1);
}
