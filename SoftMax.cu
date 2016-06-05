#include "THCUNN.h"
#include "common.h"

#define MINUS_LOG_THRESHOLD -18.42
#define SOFTMAX_THREADS 128

__global__ void cunn_SoftMax_updateOutput_kernel(
  float *output, float *input, int nframe, int dim, int stride0, int stride1)
{
  __shared__ float buffer[SOFTMAX_THREADS+1];
  float *input_k  = input  + blockIdx.x*dim*stride0 + blockIdx.y*stride1 + blockIdx.z;
  float *output_k = output + blockIdx.x*dim*stride0 + blockIdx.y*stride1 + blockIdx.z;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // max?
  buffer[threadIdx.x] = -FLT_MAX;
  for (int i=i_start; i<i_end; i+=i_step)
  {
    float z = input_k[i*stride0];
    if (buffer[threadIdx.x] < z)
      buffer[threadIdx.x] = z;
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float max_k = -FLT_MAX;
    for (int i=0; i<blockDim.x; i++)
    {
      if (max_k < buffer[i])
        max_k = buffer[i];
    }
    buffer[SOFTMAX_THREADS] = max_k;
  }

  __syncthreads();

  // sum?
  float max_k = buffer[SOFTMAX_THREADS];
  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step) {
    float z = __expf(input_k[i*stride0]-max_k);
    buffer[threadIdx.x] += z;
    output_k[i*stride0] = z;
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
    output_k[i*stride0] = output_k[i*stride0] / sum_k;
}

__global__ void cunn_SoftMax_updateGradInput_kernel(
  float *gradInput, float *output, float *gradOutput, int nframe, int dim, int stride0, int stride1)
{
  __shared__ float buffer[SOFTMAX_THREADS];
  float *gradInput_k  = gradInput  + blockIdx.x*dim*stride0 + blockIdx.y * stride1 + blockIdx.z;
  float *output_k     = output     + blockIdx.x*dim*stride0 + blockIdx.y * stride1 + blockIdx.z;
  float *gradOutput_k = gradOutput + blockIdx.x*dim*stride0 + blockIdx.y * stride1 + blockIdx.z;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // sum?
  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step)
    buffer[threadIdx.x] += gradOutput_k[i*stride0] * output_k[i*stride0];

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
    gradInput_k[i*stride0] = output_k[i*stride0] * (gradOutput_k[i*stride0] - sum_k);
}

void THNN_CudaSoftMax_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  input = THCudaTensor_newContiguous(state, input);
  THCudaTensor_resizeAs(state, output, input);
  long batchSize, dim, stride0, stride1 = 1;
  long blocksY = 1, blocksZ = 1;

  if (input->nDimension == 1)
  {
    batchSize = 1;
    dim = input->size[0];
    stride0 = 1;
  }
  else if (input->nDimension == 2)
  {
    batchSize = input->size[0];
    dim = input->size[1];
    stride0 = 1;
  }
  else if (input->nDimension == 3)
  {
    batchSize = 1;
    dim = input->size[0];
    blocksY = input->size[1];
    blocksZ = input->size[2];
    stride0 = blocksY * blocksZ;
    stride1 = blocksZ;
  }
  else if (input->nDimension == 4)
  {
    batchSize = input->size[0];
    dim = input->size[1];
    blocksY = input->size[2];
    blocksZ = input->size[3];
    stride0 = blocksY * blocksZ;
    stride1 = blocksZ;
  }
  else
  {
    THError("1D, 2D, 3D or 4D tensor expected");
  }

  // when possible use only 2d grid of thread blocks to stay compatible with compute capability 2.X devices.
  if (blocksY * blocksZ < 65536)
  {
    blocksY *= blocksZ;
    blocksZ = 1;
  }

  dim3 blocks(batchSize, blocksY, blocksZ);
  dim3 threads(SOFTMAX_THREADS);
  cunn_SoftMax_updateOutput_kernel<<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, output),
    THCudaTensor_data(state, input),
    batchSize, dim, stride0, stride1
  );
  THCudaCheck(cudaGetLastError());

  THCudaTensor_free(state, input);
}

void THNN_CudaSoftMax_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);

  output = THCudaTensor_newContiguous(state, output);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  THCudaTensor_resizeAs(state, gradInput, output);
  long batchSize, dim, stride0, stride1 = 1;
  long blocksY = 1, blocksZ = 1;

  if (gradInput->nDimension == 1)
  {
    batchSize = 1;
    dim = gradInput->size[0];
    stride0 = 1;
  }
  else if (gradInput->nDimension == 2)
  {
    batchSize = gradInput->size[0];
    dim = gradInput->size[1];
    stride0 = 1;
  }
  else if (gradInput->nDimension == 3)
  {
    batchSize = 1;
    dim = gradInput->size[0];
    blocksY = gradInput->size[1];
    blocksZ = gradInput->size[2];
    stride0 = blocksY * blocksZ;
    stride1 = blocksZ;
  }
  else if (gradInput->nDimension == 4)
  {
    batchSize = gradInput->size[0];
    dim = gradInput->size[1];
    blocksY = gradInput->size[2];
    blocksZ = gradInput->size[3];
    stride0 = blocksY * blocksZ;
    stride1 = blocksZ;
  }
  else
  {
    THError("1D, 2D, 3D or 4D tensor expected");
  }

  // when possible use only 2d grid of thread blocks to stay compatible with compute capability 2.X devices.
  if (blocksY * blocksZ < 65536)
  {
    blocksY *= blocksZ;
    blocksZ = 1;
  }

  dim3 blocks(batchSize, blocksY, blocksZ);
  dim3 threads(SOFTMAX_THREADS);
  cunn_SoftMax_updateGradInput_kernel<<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, gradInput),
    THCudaTensor_data(state, output),
    THCudaTensor_data(state, gradOutput),
    batchSize, dim, stride0, stride1
  );
  THCudaCheck(cudaGetLastError());

  THCudaTensor_free(state, gradOutput);
  THCudaTensor_free(state, output);
}
