#include "THCUNN.h"
#include "THCReduce.cuh"
#include "common.h"

#include <thrust/functional.h>

struct PReLUUpdateOutput
{
  float* weight_;

  PReLUUpdateOutput(float* weight)
    : weight_(weight)
  {}

  __device__ __forceinline__ void operator()(float *out, float *in)
  {
    float x = *in;
    *out = (x > 0) ? x : weight_[0] * x;
  }
};

__global__ void preluForward(float *output, const float *input, const float *weight, int n, int nElemsPerSample, int mapSize)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    int positionInSample = i % nElemsPerSample;
    int mapNumber = positionInSample / mapSize;
    output[i] = input[i] > 0 ? input[i] : input[i] * weight[mapNumber];
  }
}

void THNN_CudaPReLU_updateOutput(
  THCState *state,
  THCudaTensor *input,
  THCudaTensor *output,
  THCudaTensor *weight,
  long nOutputPlane)
{
  THCudaTensor_resizeAs(state, output, input);

  float *w = THCudaTensor_data(state, weight);

  if (nOutputPlane == 0)
  {
    THCudaTensor_pointwiseApply2(state, output, input, PReLUUpdateOutput(w));
  }
  else
  {
    int ndim = THCudaTensor_nDimension(state, input);
    input = THCudaTensor_newContiguous(state, input);

    int n = THCudaTensor_nElement(state, input);
    int mapSize = 1;
    if (ndim == 3)
      mapSize = (input->size[1] * input->size[2]);
    else if (ndim == 4)
      mapSize = (input->size[2] * input->size[3]);
    int nElemsPerSample = nOutputPlane * mapSize;
    preluForward<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      THCudaTensor_data(state, output),
      THCudaTensor_data(state, input),
      w,
      n, nElemsPerSample, mapSize
    );

    THCudaTensor_free(state, input);
  }
}

struct PReLUUpdateGradInput
{
  float *weight_;

  PReLUUpdateGradInput(float *weight)
    : weight_(weight)
  {}

  __device__ __forceinline__ void operator()(float *gradInput, float *gradOutput, float *input)
  {
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

void THNN_CudaPReLU_updateGradInput(
  THCState *state,
  THCudaTensor *input,
  THCudaTensor *gradOutput,
  THCudaTensor *gradInput,
  THCudaTensor *weight,
  long nOutputPlane)
{
  THCudaTensor_resizeAs(state, gradInput, input);

  float *w = THCudaTensor_data(state, weight);
  if (nOutputPlane == 0)
  {
    THCudaTensor_pointwiseApply3(state, gradInput, gradOutput, input, PReLUUpdateGradInput(w));
  }
  else
  {
    int ndim = THCudaTensor_nDimension(state, input);
    input = THCudaTensor_newContiguous(state, input);
    gradOutput = THCudaTensor_newContiguous(state, gradOutput);

    int n = THCudaTensor_nElement(state, input);
    int mapSize = 1;
    if (ndim == 3)
      mapSize = (input->size[1] * input->size[2]);
    else if (ndim == 4)
      mapSize = (input->size[2] * input->size[3]);
    int nElemsPerSample = nOutputPlane * mapSize;
    preluBackward<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      THCudaTensor_data(state, gradInput),
      THCudaTensor_data(state, input),
      w,
      THCudaTensor_data(state, gradOutput),
      n, nElemsPerSample, mapSize
    );

    THCudaTensor_free(state, input);
    THCudaTensor_free(state, gradOutput);
  }
}

struct PReLUAccGradParametersShared
{
  __device__ __forceinline__ void operator()(float *gradInput, float  *input, float *gradOutput)
  {
    *gradInput = (*input) * (*gradOutput) * (*input <= 0);
  }
};

struct PReLUAccGradParameters
{
  float scale;

  PReLUAccGradParameters(float scale)
    : scale(scale)
  {}

  __device__ __forceinline__ void operator()(float *gradInput, float *input, float *gradOutput)
  {
    *gradInput = (*input) * (*gradOutput) * scale * (*input <= 0);
  }
};

struct PReLUAccGradParameters1to1
{
  float scale;

  PReLUAccGradParameters1to1(float scale)
    : scale(scale)
  {}

  __device__ __forceinline__ void operator()(float *gradWeight, float *input, float *gradOutput)
  {
    *gradWeight += (*input) * (*gradOutput) * scale * (*input <= 0);
  }
};

void THNN_CudaPReLU_accGradParameters(
  THCState *state,
  THCudaTensor *input,
  THCudaTensor *gradOutput,
  THCudaTensor *gradInput,
  THCudaTensor *weight,
  THCudaTensor *gradWeight,
  THCudaTensor *gradWeightBuf,
  THCudaTensor *gradWeightBuf2,
  long nOutputPlane,
  float scale)
{
  // use grad input for temporary storage, then call updateGradInput again

  if (nOutputPlane == 0)
  {
    THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput, PReLUAccGradParametersShared());

    // introduces a sync point
    float sum = THCudaTensor_sumall(state, gradInput);
    float w = THCudaTensor_get1d(state, gradWeight, 0);
    THCudaTensor_set1d(state, gradWeight, 0, w + sum * scale);

    // restore gradInput
    THNN_CudaPReLU_updateGradInput(state, input, gradOutput, gradInput, weight, nOutputPlane);
  }
  else
  {
    int ndim = THCudaTensor_nDimension(state, input);

    if (ndim == 1)
    {
      THCudaTensor_pointwiseApply3(state, gradWeight, input, gradOutput, PReLUAccGradParameters1to1(scale));
    }
    else
    {
      THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput, PReLUAccGradParameters(scale));
      THCudaTensor *sumbuf = gradWeightBuf2;
      THCudaTensor_resizeAs(state, gradWeightBuf, gradWeight);

      if (ndim == 2)
      {
        THCudaTensor_sum(state, gradWeightBuf, gradInput, 0);
        THCudaTensor_cadd(state, gradWeight, gradWeight, scale, gradWeightBuf);
      }
      else if (ndim == 3)
      {
        THCudaTensor *buffer = THCudaTensor_newContiguous(state, gradInput);
        THCudaTensor_resize2d(state, buffer, nOutputPlane, input->size[1] * input->size[2]);
        THCudaTensor_sum(state, gradWeightBuf, buffer, 1);
        THCudaTensor_cadd(state, gradWeight, gradWeight, scale, gradWeightBuf);
        THCudaTensor_free(state, buffer);
      }
      else if (ndim == 4)
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
      THNN_CudaPReLU_updateGradInput(state, input, gradOutput, gradInput, weight, nOutputPlane);
    }
  }
}
