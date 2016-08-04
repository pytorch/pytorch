#include "THCUNN.h"
#include "common.h"

#define MULTIMARGIN_THREADS 128

template <int P>
__global__ void cunn_MultiMarginCriterion_updateOutput_kernel(float *output, float *input, float *target, float *weights, int nframe, int dim, bool sizeAverage, float margin)
{
  __shared__ float buffer[MULTIMARGIN_THREADS];
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *output_k = output + k;
  int target_k = ((int)target[k])-1;
  float input_target_k = input_k[target_k];

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  buffer[threadIdx.x] = 0;
  for (int i = i_start; i < i_end; i += i_step)
  {
    float z = margin - input_target_k + input_k[i];
    if (i == target_k)
      continue;

    if (z > 0) {
      float h = (P==1) ? z : z*z;
      if(weights)
        h *= weights[target_k];
      buffer[threadIdx.x] += h;
    }
  }
  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float sum = 0;
    for (int i=0; i < blockDim.x; i++)
      sum += buffer[i];

    *output_k = sum/dim;
    if(sizeAverage)
      *output_k /= nframe;
  }
}

template <int P>
__global__ void cunn_MultiMarginCriterion_updateGradInput_kernel(float *gradInput, float *input, float *target, float *weights, int nframe, int dim, bool sizeAverage, float margin)
{
  __shared__ float buffer[MULTIMARGIN_THREADS];
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *gradInput_k = gradInput + k*dim;
  int target_k = ((int)target[k])-1;
  float input_target_k = input_k[target_k];
  float g = (sizeAverage ? 1./((float)(nframe*dim)) : 1./((float)dim));

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step)
  {
    float z = margin - input_target_k + input_k[i];
    if (i == target_k)
      continue;

    if (z > 0)
    {
      float h = (P == 1) ? g : 2*g*z;
      if(weights)
        h *= weights[target_k];
      buffer[threadIdx.x] -= h;
      gradInput_k[i] = h;
    }
    else
      gradInput_k[i] = 0;
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float gradInput_target_k = 0;
    for (int i=0; i<blockDim.x; i++)
      gradInput_target_k += buffer[i];
    gradInput_k[target_k] = gradInput_target_k;
  }
}

void THNN_CudaMultiMarginCriterion_updateOutput(THCState *state, THCudaTensor *input,
                                                THCudaTensor *target, THCudaTensor *output,
                                                bool sizeAverage, int p, THCudaTensor *weights,
                                                float margin)
{
  THCUNN_assertSameGPU(state, 2, input, target);
  input = THCudaTensor_newContiguous(state, input);
  if(weights)
    weights = THCudaTensor_newContiguous(state, weights);
  if (input->nDimension == 1)
  {
    dim3 blocks(1);
    dim3 threads(MULTIMARGIN_THREADS);
    if (p == 1)
    {
      cunn_MultiMarginCriterion_updateOutput_kernel<1> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
        THCudaTensor_data(state, output),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        weights ? THCudaTensor_data(state, weights) : NULL,
        1, input->size[0],
        sizeAverage,
        margin
      );
    }
    else if (p == 2)
    {
      cunn_MultiMarginCriterion_updateOutput_kernel<2> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
        THCudaTensor_data(state, output),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        weights ? THCudaTensor_data(state, weights) : NULL,
        1, input->size[0],
        sizeAverage,
        margin
      );
    }
    THCudaCheck(cudaGetLastError());
  }
  else if (input->nDimension == 2)
  {
    THCudaTensor *output_ = THCudaTensor_newWithSize1d(state, input->size[0]);  // tmp outupt buffer
    dim3 blocks(input->size[0]);
    dim3 threads(MULTIMARGIN_THREADS);
    if (p == 1)
    {
      cunn_MultiMarginCriterion_updateOutput_kernel<1> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
        THCudaTensor_data(state, output_),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        weights ? THCudaTensor_data(state, weights) : NULL,
        input->size[0], input->size[1],
        sizeAverage,
        margin
      );
    }
    else if (p == 2)
    {
      cunn_MultiMarginCriterion_updateOutput_kernel<2> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
        THCudaTensor_data(state, output_),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        weights ? THCudaTensor_data(state, weights) : NULL,
        input->size[0], input->size[1],
        sizeAverage,
        margin
      );
    }
    THCudaCheck(cudaGetLastError());
    float sum = THCudaTensor_sumall(state, output_);
    THCudaTensor_set1d(state, output, 0, sum);
    THCudaTensor_free(state, output_);
  }
  else
  {
    THError("vector or matrix expected");
  }

  THCudaTensor_free(state, input);
  if(weights)
    THCudaTensor_free(state, weights);
}

void THNN_CudaMultiMarginCriterion_updateGradInput(THCState *state, THCudaTensor *input,
                                                   THCudaTensor *target, THCudaTensor *gradInput,
                                                   bool sizeAverage, int p, THCudaTensor *weights,
                                                   float margin)
{
  THCUNN_assertSameGPU(state, 3, input, gradInput, target);
  input = THCudaTensor_newContiguous(state, input);
  THCudaTensor_resizeAs(state, gradInput, input);
  if(weights)
    weights = THCudaTensor_newContiguous(state, weights);

  if (input->nDimension == 1)
  {
    dim3 blocks(1);
    dim3 threads(MULTIMARGIN_THREADS);

    if (p == 1)
    {
      cunn_MultiMarginCriterion_updateGradInput_kernel<1> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
        THCudaTensor_data(state, gradInput),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        weights ? THCudaTensor_data(state, weights) : NULL,
        1, gradInput->size[0],
        sizeAverage,
        margin
      );
    }
    else if (p == 2)
    {
      cunn_MultiMarginCriterion_updateGradInput_kernel<2> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
        THCudaTensor_data(state, gradInput),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        weights ? THCudaTensor_data(state, weights) : NULL,
        1, gradInput->size[0],
        sizeAverage,
        margin
      );
    }
    THCudaCheck(cudaGetLastError());
  }
  else if (input->nDimension == 2)
  {
    dim3 blocks(gradInput->size[0]);
    dim3 threads(MULTIMARGIN_THREADS);

    if (p == 1)
    {
      cunn_MultiMarginCriterion_updateGradInput_kernel<1> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
        THCudaTensor_data(state, gradInput),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        weights ? THCudaTensor_data(state, weights) : NULL,
        gradInput->size[0], gradInput->size[1],
        sizeAverage,
        margin
      );
    }
    else if (p == 2)
    {
      cunn_MultiMarginCriterion_updateGradInput_kernel<2> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
        THCudaTensor_data(state, gradInput),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        weights ? THCudaTensor_data(state, weights) : NULL,
        gradInput->size[0], gradInput->size[1],
        sizeAverage,
        margin
      );
    }
    THCudaCheck(cudaGetLastError());
  }
  else
  {
    THError("vector or matrix expected");
  }

  THCudaTensor_free(state, input);
  if(weights)
    THCudaTensor_free(state, weights);
}
