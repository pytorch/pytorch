#include "THCUNN.h"
#include "common.h"

#define MINUS_LOG_THRESHOLD -18.42
#define SOFTMAX_THREADS 128

__global__ void cunn_SoftMax_updateOutput_kernel(
  float *output, float *input, int nframe, int dim, int stride)
{
  __shared__ float buffer[SOFTMAX_THREADS+1];
  float *input_k = input + blockIdx.x*dim*stride + blockIdx.y;
  float *output_k = output + blockIdx.x*dim*stride + blockIdx.y;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // max?
  buffer[threadIdx.x] = -FLT_MAX;
  for (int i=i_start; i<i_end; i+=i_step)
  {
    float z = input_k[i*stride];
    if(buffer[threadIdx.x] < z)
      buffer[threadIdx.x] = z;
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float max_k = -FLT_MAX;
    for (int i=0; i<blockDim.x; i++)
    {
      if(max_k < buffer[i])
        max_k = buffer[i];
    }
    buffer[SOFTMAX_THREADS] = max_k;
  }

  __syncthreads();

  // sum?
  float max_k = buffer[SOFTMAX_THREADS];
  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step) {
    float z = __expf(input_k[i*stride]-max_k);
    buffer[threadIdx.x] += z;
    output_k[i*stride] = z;
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float sum_k = 0;
    for (int i=0; i<blockDim.x; i++)
      sum_k += buffer[i];
    buffer[SOFTMAX_THREADS] = sum_k;
  }

  __syncthreads();

  // softmax
  float sum_k = buffer[SOFTMAX_THREADS];
  for (int i=i_start; i<i_end; i+=i_step)
    output_k[i*stride] = output_k[i*stride] / sum_k;
}

__global__ void cunn_SoftMax_updateGradInput_kernel(
  float *gradInput, float *output, float *gradOutput, int nframe, int dim, int stride)
{
  __shared__ float buffer[SOFTMAX_THREADS];
  float *gradInput_k = gradInput + blockIdx.x*dim*stride + blockIdx.y;
  float *output_k = output + blockIdx.x*dim*stride + blockIdx.y;
  float *gradOutput_k = gradOutput + blockIdx.x*dim*stride + blockIdx.y;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // sum?
  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step)
    buffer[threadIdx.x] += gradOutput_k[i*stride] * output_k[i*stride];

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float sum_k = 0;
    for (int i=0; i<blockDim.x; i++)
      sum_k += buffer[i];
    buffer[0] = sum_k;
  }

  __syncthreads();

  float sum_k = buffer[0];
  for (int i=i_start; i<i_end; i+=i_step)
    gradInput_k[i*stride] = output_k[i*stride] * (gradOutput_k[i*stride] - sum_k);
}

void THNN_CudaSoftMax_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  input = THCudaTensor_newContiguous(state, input);
  THCudaTensor_resizeAs(state, output, input);
  long batchSize, dim, stride;

  if (input->nDimension == 1)
  {
    batchSize = 1;
    dim = input->size[0];
    stride = 1;
  }
  else if (input->nDimension == 2)
  {
    batchSize = input->size[0];
    dim = input->size[1];
    stride = 1;
  }
  else if (input->nDimension == 3)
  {
    batchSize = 1;
    dim = input->size[0];
    stride = input->size[1] * input->size[2];
  }
  else if (input->nDimension == 4)
  {
    batchSize = input->size[0];
    dim = input->size[1];
    stride = input->size[2] * input->size[3];
  }
  else
  {
    THError("1D, 2D, 3D or 4D tensor expected");
  }

  dim3 blocks(batchSize, stride);
  dim3 threads(SOFTMAX_THREADS);
  cunn_SoftMax_updateOutput_kernel<<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, output),
    THCudaTensor_data(state, input),
    batchSize, dim, stride
  );

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(state, input);
}

void THNN_CudaSoftMax_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);

  output = THCudaTensor_newContiguous(state, output);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  THCudaTensor_resizeAs(state, gradInput, output);
  long batchSize, dim, stride;

  if (gradInput->nDimension == 1)
  {
    batchSize = 1;
    dim = gradInput->size[0];
    stride = 1;
  }
  else if (gradInput->nDimension == 2)
  {
    batchSize = gradInput->size[0];
    dim = gradInput->size[1];
    stride = 1;
  }
  else if (gradInput->nDimension == 3)
  {
    batchSize = 1;
    dim = gradInput->size[0];
    stride = gradInput->size[1] * gradInput->size[2];
  }
  else if (gradInput->nDimension == 4)
  {
    batchSize = gradInput->size[0];
    dim = gradInput->size[1];
    stride = gradInput->size[2] * gradInput->size[3];
  }
  else
  {
    THError("1D, 2D, 3D or 4D tensor expected");
  }

  dim3 blocks(batchSize, stride);
  dim3 threads(SOFTMAX_THREADS);
  cunn_SoftMax_updateGradInput_kernel<<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, gradInput),
    THCudaTensor_data(state, output),
    THCudaTensor_data(state, gradOutput),
    batchSize, dim, stride
  );

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(state, gradOutput);
  THCudaTensor_free(state, output);
}
