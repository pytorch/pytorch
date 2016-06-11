#include "THCUNN.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"

#include <cfloat>

__global__ void cuda_VolumetricMaxUnpooling_updateOutput(
  THCDeviceTensor<float, 4> input,
  THCDeviceTensor<float, 4> indices,
  THCDeviceTensor<float, 4> output,
  int dT, int dH, int dW,
  int padT, int padH, int padW, int offsetZ)
{
  long iColumn = blockIdx.x * blockDim.x + threadIdx.x;
  long iRow    = blockIdx.y * blockDim.y + threadIdx.y;
  long iFrame  = (blockIdx.z + offsetZ) % input.getSize(1); // intput frame/time
  long slice   = (blockIdx.z + offsetZ) / input.getSize(1); // intput slice/feature

  if (iRow < input.getSize(2) && iColumn < input.getSize(3))
  {
    long start_t = iFrame * dT - padT;
    long start_h = iRow * dH - padH;
    long start_w = iColumn * dW - padW;

    float val = input[slice][iFrame][iRow][iColumn];
    
    float *idx = &indices[slice][iFrame][iRow][iColumn];
    long maxz = ((unsigned char*)(idx))[0];
    long maxy = ((unsigned char*)(idx))[1];
    long maxx = ((unsigned char*)(idx))[2];
    output[slice][start_t + maxz][start_h + maxy][start_w + maxx] = val;
  }
}

void THNN_CudaVolumetricMaxUnpooling_updateOutput(
  THCState *state, THCudaTensor *input, THCudaTensor *output, THCudaTensor *indices,
  int outputTime, int outputWidth, int outputHeight,
  int dT, int dW, int dH,
  int padT, int padW, int padH)
{
  int batchSize;
  int inputSlices;
  int inputTime;
  int inputHeight;
  int inputWidth;

  THCUNN_assertSameGPU(state, 3, input, indices, output);

  if (THCudaTensor_nDimension(state, input) == 4)
  {
    /* sizes */
    batchSize   = 1;
    inputSlices = THCudaTensor_size(state, input, 0);
    inputTime   = THCudaTensor_size(state, input, 1);
    inputHeight = THCudaTensor_size(state, input, 2);
    inputWidth  = THCudaTensor_size(state, input, 3);
  }
  else if (THCudaTensor_nDimension(state, input) == 5)
  {
    /* sizes */
    batchSize   = THCudaTensor_size(state, input, 0);
    inputSlices = THCudaTensor_size(state, input, 1);
    inputTime   = THCudaTensor_size(state, input, 2);
    inputHeight = THCudaTensor_size(state, input, 3);
    inputWidth  = THCudaTensor_size(state, input, 4);
  }
  else
  {
    THArgCheck(false, 2, "4D or 5D tensor expected");
  }

  if (input->nDimension == 4) /* 4D */
  {
    /* resize output */
    THCudaTensor_resize4d(state, output, inputSlices,
                          outputTime, outputHeight, outputWidth);
  }
  else
  { /* 5D */
    THCudaTensor_resize5d(state, output, batchSize, inputSlices,
                          outputTime, outputHeight, outputWidth);
  }

  input = THCudaTensor_newContiguous(state, input);
  indices = THCudaTensor_newContiguous(state, indices);
  THCudaTensor_zero(state, output);
  
  // Collapse batch and feature dimensions
  THCDeviceTensor<float, 4> cudaInput;
  THCDeviceTensor<float, 4> cudaOutput;
  THCDeviceTensor<float, 4> cudaIndices;

  if (THCudaTensor_nDimension(state, input) == 4)
  {
    cudaInput  = toDeviceTensor<float, 4>(state, input);
    cudaOutput = toDeviceTensor<float, 4>(state, output);
    cudaIndices = toDeviceTensor<float, 4>(state, indices);
  }
  else
  {
    cudaInput  = toDeviceTensor<float, 5>(state, input).downcastOuter<4>();
    cudaOutput = toDeviceTensor<float, 5>(state, output).downcastOuter<4>();
    cudaIndices = toDeviceTensor<float, 5>(state, indices).downcastOuter<4>();
  }

  int totalZ = inputTime * inputSlices * batchSize;
  int offsetZ = 0;
  dim3 block(32, 8);

  while (totalZ > 0) {
    dim3 grid(THCCeilDiv(inputWidth, static_cast<int>(block.x)),
              THCCeilDiv(inputHeight, static_cast<int>(block.y)),
              totalZ > 65535 ? 65535 : totalZ);

    cuda_VolumetricMaxUnpooling_updateOutput<<<grid, block,
          0, THCState_getCurrentStream(state)>>>(
                             cudaInput, cudaIndices, cudaOutput,
                             dT, dH, dW,
                             padT, padH, padW, offsetZ);
    THCudaCheck(cudaGetLastError());
    totalZ -= 65535;
    offsetZ += 65535;
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, indices);
}

__global__ void cuda_VolumetricMaxUnpooling_updateGradInput(
  THCDeviceTensor<float, 4> gradOutput,
  THCDeviceTensor<float, 4> indices,
  THCDeviceTensor<float, 4> gradInput,
  int dT, int dH, int dW,
  int padT, int padH, int padW, int offsetZ)
{
  int iColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int iRow    = blockIdx.y * blockDim.y + threadIdx.y;
  int iFrame  = (blockIdx.z + offsetZ) % gradInput.getSize(1); // output frame/time
  int slice   = (blockIdx.z + offsetZ) / gradInput.getSize(1); // output slice/feature

  if (iRow < gradInput.getSize(2) && iColumn < gradInput.getSize(3))
  {
    
    long start_t = iFrame * dT - padT;
    long start_h = iRow * dH - padH;
    long start_w = iColumn * dW - padW;

    float *idx = &indices[slice][iFrame][iRow][iColumn];
    long maxz = ((unsigned char*)(idx))[0];
    long maxy = ((unsigned char*)(idx))[1];
    long maxx = ((unsigned char*)(idx))[2];

    float grad_val = gradOutput[slice][start_t + maxz][start_h + maxy][start_w + maxx];

    gradInput[slice][iFrame][iRow][iColumn] = grad_val;
  }
}

void THNN_CudaVolumetricMaxUnpooling_updateGradInput(
  THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput,
  THCudaTensor *indices,
  int outputTime, int outputWidth, int outputHeight,
  int dT, int dW, int dH,
  int padT, int padW, int padH)
{
  
  int batchSize;
  int inputSlices;
  int inputTime;
  int inputHeight;
  int inputWidth;
  
  THCUNN_assertSameGPU(state, 4, input, indices, gradOutput, gradInput);

  if (THCudaTensor_nDimension(state, input) == 4) /* 4D */
  {
    batchSize = 1;
    inputSlices  = THCudaTensor_size(state, input, 0);
    inputTime   = THCudaTensor_size(state, input, 1);
    inputHeight = THCudaTensor_size(state, input, 2);
    inputWidth  = THCudaTensor_size(state, input, 3);
  }
  else
  {
    batchSize    = THCudaTensor_size(state, input, 0);
    inputSlices  = THCudaTensor_size(state, input, 1);
    inputTime   = THCudaTensor_size(state, input, 2);
    inputHeight = THCudaTensor_size(state, input, 3);
    inputWidth  = THCudaTensor_size(state, input, 4);
  }

  input = THCudaTensor_newContiguous(state, input);
  indices = THCudaTensor_newContiguous(state, indices);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_zero(state, gradInput);

  // Collapse batch and feature dimensions
  THCDeviceTensor<float, 4> cudaGradInput;
  THCDeviceTensor<float, 4> cudaGradOutput;
  THCDeviceTensor<float, 4> cudaIndices;
  
  if (THCudaTensor_nDimension(state, input) == 4)
  {
    cudaGradInput  = toDeviceTensor<float, 4>(state, gradInput);
    cudaGradOutput = toDeviceTensor<float, 4>(state, gradOutput);
    cudaIndices = toDeviceTensor<float, 4>(state, indices);
  }
  else
  {
    cudaGradInput =
      toDeviceTensor<float, 5>(state, gradInput).downcastOuter<4>();
    cudaGradOutput =
      toDeviceTensor<float, 5>(state, gradOutput).downcastOuter<4>();
    cudaIndices =
      toDeviceTensor<float, 5>(state, indices).downcastOuter<4>();
  }

  int totalZ = inputTime * inputSlices * batchSize;
  int offsetZ = 0;
  dim3 block(32, 8);

  while (totalZ > 0) {
    dim3 grid(THCCeilDiv(inputWidth, static_cast<int>(block.x)),
              THCCeilDiv(inputHeight, static_cast<int>(block.y)),
              totalZ > 65535 ? 65535 : totalZ);

    cuda_VolumetricMaxUnpooling_updateGradInput<<<grid, block,
      0, THCState_getCurrentStream(state)>>>(
                                             cudaGradOutput,
                                             cudaIndices,
                                             cudaGradInput,
                                             dT, dH, dW,
                                             padT, padH, padW, offsetZ);
    THCudaCheck(cudaGetLastError());
    totalZ -= 65535;
    offsetZ += 65535;
  }

  // cleanup
  THCudaTensor_free(state, input);
  THCudaTensor_free(state, gradOutput);
  THCudaTensor_free(state, indices);
}