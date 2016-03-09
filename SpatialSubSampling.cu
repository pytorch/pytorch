#include "THCUNN.h"
#include "common.h"

#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit

/*
 * Description:
 *    this function subsamples an input 3D tensor along dimensions 1 and 2
 *    3D input, 3D output, 1D weight, 1D bias
 */
__global__ void subsample(float *input, float *output, float *weight, float *bias,
                          int input_n, int input_h, int input_w,
                          int kH, int kW, int dH, int dW)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;
  int k = blockIdx.x % input_n;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  output = output + o*output_w*output_h;
  input = input + i*input_w*input_h;

  // Get the good mask for (k,i) (k out, i in)
  float the_weight = weight[k];

  // Initialize to the bias
  float the_bias = bias[k];

  // For all output pixels...
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      // Compute the mean of the input image...
      float *ptr_input = input + yy*dH*input_w + xx*dW;
      float *ptr_output = output + yy*output_w + xx;
      float sum = 0;
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++)
          sum += ptr_input[kx];
        ptr_input += input_w; // next input line
      }
      // Update output
      *ptr_output = the_weight*sum + the_bias;
    }
  }
}

/*
 * Description:
 *    this function computes the gradWeight from input and gradOutput
 */
__global__ void subgradweight(float *input, float *gradOutput, float *gradWeight, float *gradBias,
                              int input_n, int input_h, int input_w,
                              int kH, int kW, int dH, int dW,
                              float scale)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;
  int k = blockIdx.x % input_n;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  input = input + i*input_w*input_h;

  // thread ID
  int tid = blockDim.x*threadIdx.y + threadIdx.x;

  // create array to hold partial sums
  __shared__ float sums[CUDA_MAX_THREADS];
  sums[tid] = 0;

  // compute partial sums
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      float *ptr_input = input + yy*dH*input_w + xx*dW;
      float *ptr_gradOutput = gradOutput + yy*output_w + xx;
      float z = *ptr_gradOutput;
      long kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++) {
          sums[tid] += z * ptr_input[kx];
        }
        ptr_input += input_w;
      }
    }
  }
  __syncthreads();

  // reduce: accumulate all partial sums to produce final gradWeight
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for(int i = 0; i < blockDim.x*blockDim.y; i++) gradWeight[k] += scale*sums[i];
  }
  __syncthreads();

  // compute gradBias
  sums[tid] = 0;
  for (int i=tid; i<output_w*output_h; i+=(blockDim.x*blockDim.y)) {
    sums[tid] += gradOutput[i];
  }
  __syncthreads();

  // reduce gradBias
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i=0; i<(blockDim.x*blockDim.y); i++)
      gradBias[k] += scale*sums[i];
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 */
__global__ void subgradinput(float *gradInput, float *gradOutput, float *weight,
                             int input_n, int input_h, int input_w,
                             int kH, int kW, int dH, int dW)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;
  int k = blockIdx.x % input_n;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  gradInput = gradInput + i*input_w*input_h;

  // get weight
  float the_weight = weight[k];

  // compute gradInput
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
      float *ptr_gradOutput = gradOutput + yy*output_w + xx;
      float z = *ptr_gradOutput * the_weight;
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++)
          ptr_gradInput[kx] += z;
        ptr_gradInput += input_w;
      }
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 */
__global__ void subgradinputAtomic(float *gradInput, float *gradOutput, float *weight,
                                   int input_n, int input_h, int input_w,
                                   int kH, int kW, int dH, int dW)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;
  int k = blockIdx.x % input_n;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  gradInput = gradInput + i*input_w*input_h;

  // get weight
  float the_weight = weight[k];

  // compute gradInput
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
      float *ptr_gradOutput = gradOutput + yy*output_w + xx;
      float z = *ptr_gradOutput * the_weight;
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++) {
          atomicAdd(&(ptr_gradInput[kx]), z);
        }
        ptr_gradInput += input_w;
      }
    }
  }
}

void THNN_CudaSpatialSubSampling_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output, THCudaTensor *weight, THCudaTensor *bias, int kW, int kH, int dW, int dH)
{
  float *weight_data = THCudaTensor_data(state, weight);
  float *bias_data = THCudaTensor_data(state, bias);
  float *output_data;
  float *input_data;

  int nInputPlane = THCudaTensor_size(state, weight, 0);

  THCUNN_assertSameGPU(state, 4, input, output, weight, bias);
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;

    THArgCheck(input->size[0] == nInputPlane, 2, "invalid number of input planes");
    THArgCheck(nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THCudaTensor_newContiguous(state, input);
    input_data = THCudaTensor_data(state, input);

    THCudaTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);
    output_data = THCudaTensor_data(state, output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    // run subsample kernel
    subsample <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
      input_data, output_data, weight_data, bias_data,
      nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nbatch = input->size[0];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;

    THArgCheck(input->size[1] == nInputPlane, 2, "invalid number of input planes");
    THArgCheck(nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THCudaTensor_newContiguous(state, input);
    input_data = THCudaTensor_data(state, input);

    THCudaTensor_resize4d(state, output, nbatch, nInputPlane, nOutputRows, nOutputCols);
    output_data = THCudaTensor_data(state, output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    // run subsample kernel
    subsample <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
      input_data, output_data, weight_data, bias_data,
      nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
  }

  // clean
  THCudaTensor_free(state, input);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialSubsampling.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}

void THNN_CudaSpatialSubSampling_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput, THCudaTensor *weight, int kW, int kH, int dW, int dH)
{
  THCUNN_assertSameGPU(state, 4, input, gradOutput, weight, gradInput);

  int nInputPlane = THCudaTensor_size(state, weight, 0);

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];

    float *weight_data = THCudaTensor_data(state, weight);
    float *gradOutput_data = THCudaTensor_data(state, gradOutput);
    float *gradInput_data;

    THCudaTensor_resizeAs(state, gradInput, input);
    THCudaTensor_zero(state, gradInput);
    gradInput_data = THCudaTensor_data(state, gradInput);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    // run updateGradInput kernel
    subgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
      gradInput_data, gradOutput_data, weight_data,
      nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nbatch = input->size[0];

    float *weight_data = THCudaTensor_data(state, weight);
    float *gradOutput_data = THCudaTensor_data(state, gradOutput);
    float *gradInput_data;

    THCudaTensor_resizeAs(state, gradInput, input);
    THCudaTensor_zero(state, gradInput);
    gradInput_data = THCudaTensor_data(state, gradInput);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    // run updateGradInput kernel
    if (kH == dH && kW == dW) {
      subgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
        gradInput_data, gradOutput_data, weight_data,
        nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
    } else {
      subgradinputAtomic <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
        gradInput_data, gradOutput_data, weight_data,
        nInputPlane, nInputRows, nInputCols,
        kH, kW, dH, dW);
    }
  }

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialSubsampling.updateGradInput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}

void THNN_CudaSpatialSubSampling_accGradParameters(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradWeight, THCudaTensor *gradBias, int kW, int kH, int dW, int dH, float scale)
{
  THCUNN_assertSameGPU(state, 4, input, gradOutput, gradWeight, gradBias);

  int nInputPlane = THCudaTensor_size(state, gradWeight, 0);

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];

    float *gradWeight_data = THCudaTensor_data(state, gradWeight);
    float *gradBias_data = THCudaTensor_data(state, gradBias);
    float *gradOutput_data = THCudaTensor_data(state, gradOutput);
    float *input_data;

    input = THCudaTensor_newContiguous(state, input);
    input_data = THCudaTensor_data(state, input);

    // cuda blocks & threads:
    dim3 blocks(nInputPlane);
    dim3 threads(32,8);

    // run gradweight kernel
    subgradweight <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
      input_data, gradOutput_data, gradWeight_data, gradBias_data,
      nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW, scale);
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nbatch = input->size[0];

    float *gradWeight_data = THCudaTensor_data(state, gradWeight);
    float *gradBias_data = THCudaTensor_data(state, gradBias);
    float *gradOutput_data = THCudaTensor_data(state, gradOutput);
    float *input_data;

    input = THCudaTensor_newContiguous(state, input);
    input_data = THCudaTensor_data(state, input);

    // cuda blocks & threads:
    dim3 blocks(nInputPlane);
    dim3 threads(32,8);

    // run gradweight kernel
    long sl;
    for (sl=0; sl<nbatch; sl++) {
      subgradweight <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
        input_data + sl*input->stride[0],
        gradOutput_data + sl*gradOutput->stride[0],
        gradWeight_data, gradBias_data,
        nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW, scale);
    }
  }

  // clean
  THCudaTensor_free(state, input);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialSubsampling.accGradParameters: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}

#undef CUDA_MAX_THREADS
