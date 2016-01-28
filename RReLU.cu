#include "THCApply.cuh"
#include "utils.h"
#include "common.h"
#include <curand.h>
#include <curand_kernel.h>

// copied from cutorch/lib/THC/THCTensorRandom.cu
#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256
#define NUM_BLOCKS(n) min((int)THCCeilDiv(n, (long) BLOCK_SIZE), MAX_NUM_BLOCKS)

__global__ void rreluUpdateOutputTrain(int n, curandStateMtgp32 *state, 
  float *input, float* noise, float *output, double a, double b)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    if (input[i] <= 0)
    {
      float r = curand_uniform(&state[blockIdx.x]);
      r = r * (b-a) + a;
      output[i] = input[i] * r;
      noise[i] = r;
    }
    else
    {
      output[i] = input[i];
      noise[i] = 1;
    }
  }
}

struct RReLUUpdateOutputEval_functor
{
  const float negSlope_;

  RReLUUpdateOutputEval_functor(float negSlope) : negSlope_(negSlope) {}

  __device__ __forceinline__ void operator()(float* out, float* in)
  {
    const float x = *in;
    const float r = x <= 0 ? negSlope_ : 1;
    *out = x * r;
  }
};

struct RReLUUpdateOutputEvalIP_functor
{
  const float negSlope_;

  RReLUUpdateOutputEvalIP_functor(float negSlope) : negSlope_(negSlope) {}

  __device__ __forceinline__ void operator()(float* x)
  {
    if (*x <= 0)
    {
      *x = *x * negSlope_;
    }
  }
};

static int cunn_RReLU_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *noise = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "noise", "torch.CudaTensor");
  double lower = luaT_getfieldchecknumber(L, 1, "lower");
  double upper = luaT_getfieldchecknumber(L, 1, "upper");
  int train = luaT_getfieldcheckboolean(L, 1, "train");
  int inplace = luaT_getfieldcheckboolean(L, 1, "inplace");

  THAssert(THCudaTensor_checkGPU(state, 3, input, output, noise));
  if (state->rngState->current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }

  if (train)
  {
    input = THCudaTensor_newContiguous(state, input);
    THCudaTensor_resizeAs(state, noise, input);
    float *input_data = THCudaTensor_data(state, input);
    float *noise_data = THCudaTensor_data(state, noise);
    long n = THCudaTensor_nElement(state, input);
    if (inplace)
    {
      rreluUpdateOutputTrain<<<NUM_BLOCKS(n), BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
        n, state->rngState->current_gen->gen_states, 
        input_data, noise_data, input_data, lower, upper);
      THCudaTensor_set(state, output, input);
    }
    else
    {
      THCudaTensor_resizeAs(state, output, input);
      float *output_data = THCudaTensor_data(state, output);
      rreluUpdateOutputTrain<<<NUM_BLOCKS(n), BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
        n, state->rngState->current_gen->gen_states, 
        input_data, noise_data, output_data, lower, upper);
    }
    THCudaTensor_free(state, input);
  }
  else
  {
    const double negSlope = (lower + upper) / 2;
    if (inplace)
    {
      THCudaTensor_pointwiseApply1(state, input, RReLUUpdateOutputEvalIP_functor(negSlope));
      THCudaTensor_set(state, output, input);
    }
    else
    {
      THCudaTensor_resizeAs(state, output, input);
      THCudaTensor_pointwiseApply2(state, output, input, RReLUUpdateOutputEval_functor(negSlope));
    }
  }

  return 1;
}

struct RReLUupdateGradInputEval_functor
{
  const float negSlope_;

  RReLUupdateGradInputEval_functor(float negSlope) : negSlope_(negSlope) {}

  __device__ __forceinline__ void operator()(float *gradIn, float *gradOut, float* in)
  {
    *gradIn = (*in) <= 0 ? (*gradOut) * negSlope_ : (*gradOut);
  }
};

struct RReLUupdateGradInputEvalIP_functor
{
  const float negSlope_;

  RReLUupdateGradInputEvalIP_functor(float negSlope) : negSlope_(negSlope) {}

  __device__ __forceinline__ void operator()(float *gradOut, float *in)
  {
    if (*in <= 0)
    {
      *gradOut = (*gradOut) * negSlope_;
    }
  }
};

static int cunn_RReLU_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *noise = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "noise", "torch.CudaTensor");
  double lower = luaT_getfieldchecknumber(L, 1, "lower");
  double upper = luaT_getfieldchecknumber(L, 1, "upper");
  int train = luaT_getfieldcheckboolean(L, 1, "train");
  int inplace = luaT_getfieldcheckboolean(L, 1, "inplace");
  
  THAssert(THCudaTensor_checkGPU(state, 4, input, gradOutput, gradInput, noise));
  
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  
  if (train && upper - lower > 1E-6)    // e.g. if upper == lower, RReLU behaves like LeakyReLU
  {
    // multiply the gradient by the noise tensor
    if (inplace)
    {
      THCudaTensor_cmul(state, gradOutput, gradOutput, noise);
      THCudaTensor_set(state, gradInput, gradOutput);
    }
    else
    {
      THCudaTensor_resizeAs(state, gradInput, input);
      THCudaTensor_cmul(state, gradInput, gradOutput, noise);
    }    
  }
  else
  {
    // use constant factor for negative input values
    const double negSlope = (lower + upper) / 2;
    if (inplace)
    {
      THCudaTensor_pointwiseApply2(state, gradOutput, input, RReLUupdateGradInputEvalIP_functor(negSlope));
      THCudaTensor_set(state, gradInput, gradOutput);
    }
    else
    {
      THCudaTensor_resizeAs(state, gradInput, input);
      THCudaTensor_pointwiseApply3(state, gradInput, gradOutput, input, RReLUupdateGradInputEval_functor(negSlope));
    }
  }
  
  THCudaTensor_free(state, gradOutput);
  return 1;
}

static const struct luaL_Reg cunn_RReLU__ [] = {
  {"RReLU_updateOutput", cunn_RReLU_updateOutput},
  {"RReLU_updateGradInput", cunn_RReLU_updateGradInput},
  {NULL, NULL}
};

void cunn_RReLU_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_RReLU__, "nn");
  lua_pop(L,1);
}
