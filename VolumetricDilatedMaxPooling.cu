#include "THCUNN.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"

#include <cfloat>

__global__ void cuda_VolumetricDilatedMaxPooling_updateOutput(
  THCDeviceTensor<float, 4> input,
  THCDeviceTensor<THCIndex_t, 4> indices,
  THCDeviceTensor<float, 4> output,
  int kT, int kH, int kW,
  int dT, int dH, int dW,
  int padT, int padH, int padW,
  int dilationT, int dilationH, int dilationW,
  int offsetZ)
{
  int oColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int oRow    = blockIdx.y * blockDim.y + threadIdx.y;
  int oFrame  = (blockIdx.z + offsetZ) % output.getSize(1); // output frame/time
  int slice   = (blockIdx.z + offsetZ) / output.getSize(1); // output slice/feature

  if (oRow < output.getSize(2) && oColumn < output.getSize(3))
  {
    int iColumn = oColumn * dW - padW;
    int iRow    = oRow    * dH - padH;
    int iFrame  = oFrame  * dT - padT;

    int maxColumn = 0;
    int maxRow = 0;
    int maxFrame = 0;

    float max = -FLT_MAX;

    for (int frame = 0; frame < kT; ++frame)
    {
      if (iFrame + frame * dilationT < input.getSize(1) && iFrame + frame * dilationT >= 0)
      {
        for (int row = 0; row < kH; ++row)
        {
          if (iRow + row * dilationH < input.getSize(2) && iRow + row * dilationH >= 0)
          {
            for (int column = 0; column < kW; ++column)
            {
              if (iColumn + column * dilationW < input.getSize(3) && iColumn + column * dilationW >= 0)
              {
                float val = input[slice][iFrame + frame * dilationT][iRow + row * dilationH][iColumn + column * dilationW];

                if (max < val)
                {
                  max = val;
                  maxColumn = column;
                  maxRow    = row;
                  maxFrame  = frame;
                }
              }
            }
          }
        }
      }
    }

    output[slice][oFrame][oRow][oColumn] = max;
    THCIndex_t *idx = &indices[slice][oFrame][oRow][oColumn];
    ((unsigned char*)(idx))[0] = maxFrame;
    ((unsigned char*)(idx))[1] = maxRow;
    ((unsigned char*)(idx))[2] = maxColumn;
    ((unsigned char*)(idx))[3] = 0;
  }
}

template <int KERNEL_WIDTH>
__global__ void cuda_VolumetricDilatedMaxPooling_updateOutput(
  THCDeviceTensor<float, 4> input, THCDeviceTensor<THCIndex_t, 4> indices,
  THCDeviceTensor<float, 4> output,
  int kT, int kH,
  int dT, int dH, int dW,
  int padT, int padH, int padW,
  int dilationT, int dilationH, int dilationW,
  int offsetZ)
{
  int oColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int oRow    = blockIdx.y * blockDim.y + threadIdx.y;
  int oFrame  = (blockIdx.z + offsetZ) % output.getSize(1); // output frame/time
  int slice   = (blockIdx.z + offsetZ) / output.getSize(1); // output slice/feature

  if (oRow < output.getSize(2) && oColumn < output.getSize(3))
  {
    int iColumn = oColumn * dW - padW;
    int iRow    = oRow    * dH - padH;
    int iFrame  = oFrame  * dT - padT;

    int maxColumn = 0;
    int maxRow = 0;
    int maxFrame;

    float max = -FLT_MAX;

    for (int frame = 0; frame < kT; ++frame)
    {
      if (iFrame + frame * dilationT < input.getSize(1) && iFrame + frame * dilationT >= 0)
      {
        for (int row = 0; row < kH; ++row)
        {
          if (iRow + row * dilationH < input.getSize(2) && iRow + row * dilationH >= 0)
          {
            for (int column = 0; column < KERNEL_WIDTH; ++column)
            {
              if (iColumn + column * dilationW < input.getSize(3) && iColumn + column * dilationW >= 0)
              {
                float val = input[slice][iFrame + frame * dilationT][iRow + row * dilationH][iColumn + column * dilationW];

                if (max < val)
                {
                  max = val;
                  maxColumn = column;
                  maxRow    = row;
                  maxFrame  = frame;
                }
              }
            }
          }
        }
      }
    }

    output[slice][oFrame][oRow][oColumn] = max;
    THCIndex_t *idx = &indices[slice][oFrame][oRow][oColumn];
    ((unsigned char*)(idx))[0] = maxFrame;
    ((unsigned char*)(idx))[1] = maxRow;
    ((unsigned char*)(idx))[2] = maxColumn;
    ((unsigned char*)(idx))[3] = 0;
  }
}

#define UPDATE_OUTPUT_KERNEL_WIDTH(KW) case KW:                         \
  cuda_VolumetricDilatedMaxPooling_updateOutput<KW><<<grid, block,             \
    0, THCState_getCurrentStream(state)>>>(                             \
    cudaInput, cudaIndices, cudaOutput, kT, kH, dT, dH, dW, padT, padH, padW,\
    dilationT, dilationH, dilationW, offsetZ); \
    break


void THNN_CudaVolumetricDilatedMaxPooling_updateOutput(
  THCState *state, THCudaTensor *input, THCudaTensor *output, THCIndexTensor *indices,
  int kT, int kW, int kH,
  int dT, int dW, int dH,
  int padT, int padW, int padH,
  int dilationT, int dilationW, int dilationH,
  bool ceilMode)
{
  int batchSize;
  int inputSlices;
  int inputTime;
  int inputHeight;
  int inputWidth;
  int outputTime;
  int outputHeight;
  int outputWidth;

  THCUNN_assertSameGPU(state, 3, input, indices, output);

  if (THCudaTensor_nDimension(state, input) == 4)
  {
    THArgCheck(
      THCudaTensor_size(state, input, 1) >= kT &&
      THCudaTensor_size(state, input, 2) >= kH &&
      THCudaTensor_size(state, input, 3) >= kW, 2,
      "input image smaller than kernel size"
    );

    /* sizes */
    batchSize   = 1;
    inputSlices = THCudaTensor_size(state, input, 0);
    inputTime   = THCudaTensor_size(state, input, 1);
    inputHeight = THCudaTensor_size(state, input, 2);
    inputWidth  = THCudaTensor_size(state, input, 3);
  }
  else if (THCudaTensor_nDimension(state, input) == 5)
  {
    THArgCheck(
      THCudaTensor_size(state, input, 4) >= kW &&
      THCudaTensor_size(state, input, 3) >= kH &&
      THCudaTensor_size(state, input, 2) >= kT, 2,
      "input image smaller than kernel size"
    );

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

  THArgCheck(kT/2 >= padT && kW/2 >= padW && kH/2 >= padH, 2,
    "pad should be smaller than half of kernel size"
  );

  if (ceilMode)
  {
    outputTime   = (int)(ceil((float)(inputTime - (dilationT * (kT - 1) + 1) + 2*padT) / dT)) + 1;
    outputHeight = (int)(ceil((float)(inputHeight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
    outputWidth  = (int)(ceil((float)(inputWidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
  }
  else
  {
    outputTime   = (int)(floor((float)(inputTime - (dilationT * (kT - 1) + 1) + 2*padT) / dT)) + 1;
    outputHeight = (int)(floor((float)(inputHeight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
    outputWidth  = (int)(floor((float)(inputWidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
  }

  if (outputTime < 1 || outputHeight < 1 || outputWidth < 1)
    THError("Given input size: (%dx%dx%dx%d). Calculated output size: (%dx%dx%dx%d). Output size is too small",
            inputSlices,inputTime,inputHeight,inputWidth,inputSlices,outputTime,outputHeight,outputWidth);

  if (padT || padW || padH)
  {
    if ((outputTime - 1)*dT >= inputTime + padT)
      --outputTime;
    if ((outputHeight - 1)*dH >= inputHeight + padH)
      --outputHeight;
    if ((outputWidth  - 1)*dW >= inputWidth  + padW)
      --outputWidth;
  }

  if (input->nDimension == 4) /* 4D */
  {
    /* resize output */
    THCudaTensor_resize4d(state, output, inputSlices,
                          outputTime, outputHeight, outputWidth);
    /* indices pack ti,i,j locations for each output point as uchar into
     each float of the tensor */
    THCIndexTensor_(resize4d)(state, indices, inputSlices,
                          outputTime, outputHeight, outputWidth);
  }
  else
  { /* 5D */
    THCudaTensor_resize5d(state, output, batchSize, inputSlices,
                          outputTime, outputHeight, outputWidth);
    // Index tensor packs index offsets as uchars into floats
    THCIndexTensor_(resize5d)(state, indices, batchSize, inputSlices,
                          outputTime, outputHeight, outputWidth);
  }

  input = THCudaTensor_newContiguous(state, input);

  // Collapse batch and feature dimensions
  THCDeviceTensor<float, 4> cudaInput;
  THCDeviceTensor<float, 4> cudaOutput;
  if (THCudaTensor_nDimension(state, input) == 4)
  {
    cudaInput  = toDeviceTensor<float, 4>(state, input);
    cudaOutput = toDeviceTensor<float, 4>(state, output);
  }
  else
  {
    cudaInput  = toDeviceTensor<float, 5>(state, input).downcastOuter<4>();
    cudaOutput = toDeviceTensor<float, 5>(state, output).downcastOuter<4>();
  }

  THLongStorage *indicesSize = THLongStorage_newWithSize(4);
  long indicesSizeRaw[4] = { batchSize * inputSlices,
                            outputTime, outputHeight, outputWidth };
  THLongStorage_rawCopy(indicesSize, indicesSizeRaw);

  THCIndexTensor *indices1 = THCIndexTensor_(newWithStorage)(
    state, THCIndexTensor_(storage)(state, indices),
    THCIndexTensor_(storageOffset)(state, indices),
    indicesSize, NULL);

  THLongStorage_free(indicesSize);

  THCDeviceTensor<THCIndex_t, 4> cudaIndices =
    toDeviceTensor<THCIndex_t, 4>(state, indices1);

  int totalZ = outputTime * inputSlices * batchSize;
  int offsetZ = 0;
  dim3 block(32, 8);

  while (totalZ > 0) {
    dim3 grid(THCCeilDiv(outputWidth, static_cast<int>(block.x)),
              THCCeilDiv(outputHeight, static_cast<int>(block.y)),
              totalZ > 65535 ? 65535 : totalZ);

    switch (kW)
      {
        UPDATE_OUTPUT_KERNEL_WIDTH(1);
        UPDATE_OUTPUT_KERNEL_WIDTH(2);
        UPDATE_OUTPUT_KERNEL_WIDTH(3);
        UPDATE_OUTPUT_KERNEL_WIDTH(4);
        UPDATE_OUTPUT_KERNEL_WIDTH(5);
        UPDATE_OUTPUT_KERNEL_WIDTH(6);
        UPDATE_OUTPUT_KERNEL_WIDTH(7);
      default:
        cuda_VolumetricDilatedMaxPooling_updateOutput<<<grid, block,
          0, THCState_getCurrentStream(state)>>>(
                             cudaInput, cudaIndices, cudaOutput,
                             kT, kH, kW, dT, dH, dW,
                             padT, padH, padW, dilationT, dilationH, dilationW, offsetZ);
      }
    THCudaCheck(cudaGetLastError());
    totalZ -= 65535;
    offsetZ += 65535;
  }

  THCudaTensor_free(state, input);
  THCIndexTensor_(free)(state, indices1);
}

#undef UPDATE_OUTPUT_KERNEL_WIDTH

__global__ void cuda_VolumetricDilatedMaxPooling_updateGradInput(
  THCDeviceTensor<float, 4> gradOutput,
  THCDeviceTensor<THCIndex_t, 4> indices,
  THCDeviceTensor<float, 4> gradInput,
  int dT, int dH, int dW,
  int padT, int padH, int padW,
  int dilationT, int dilationH, int dilationW,
  int offsetZ)
{
  int oColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int oRow    = blockIdx.y * blockDim.y + threadIdx.y;
  int oFrame  = (blockIdx.z + offsetZ) % gradOutput.getSize(1); // output frame/time
  int slice   = (blockIdx.z + offsetZ) / gradOutput.getSize(1); // output slice/feature

  if (oRow < gradOutput.getSize(2) && oColumn < gradOutput.getSize(3))
  {
    THCIndex_t *idx = &indices[slice][oFrame][oRow][oColumn];
    int iFrame  = ((unsigned char*)(idx))[0] * dilationT + oFrame  * dT - padT;
    int iRow    = ((unsigned char*)(idx))[1] * dilationH + oRow    * dH - padH;
    int iColumn = ((unsigned char*)(idx))[2] * dilationW + oColumn * dW - padW;
    atomicAdd(&gradInput[slice][iFrame][iRow][iColumn],
              gradOutput[slice][oFrame][oRow][oColumn]);
  }
}

void THNN_CudaVolumetricDilatedMaxPooling_updateGradInput(
  THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput,
  THCIndexTensor *indices,
  int dT, int dW, int dH,
  int padT, int padW, int padH,
  int dilationT, int dilationW, int dilationH)
{
  // Resize and initialize result tensor.
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_zero(state, gradInput);

  int batchSize;
  int inputSlices;

  int outputTime;
  int outputHeight;
  int outputWidth;

  THCUNN_assertSameGPU(state, 4, input, indices, gradOutput, gradInput);

  if (THCudaTensor_nDimension(state, input) == 4) /* 4D */
  {
    batchSize = 1;
    inputSlices  = THCudaTensor_size(state, input, 0);

    outputTime   = THCudaTensor_size(state, gradOutput, 1);
    outputHeight = THCudaTensor_size(state, gradOutput, 2);
    outputWidth  = THCudaTensor_size(state, gradOutput, 3);
  }
  else
  {
    batchSize    = THCudaTensor_size(state, input, 0);
    inputSlices  = THCudaTensor_size(state, input, 1);

    outputTime   = THCudaTensor_size(state, gradOutput, 2);
    outputHeight = THCudaTensor_size(state, gradOutput, 3);
    outputWidth  = THCudaTensor_size(state, gradOutput, 4);
  }

  gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  // Collapse batch and feature dimensions
  THCDeviceTensor<float, 4> cudaGradInput;
  THCDeviceTensor<float, 4> cudaGradOutput;
  if (THCudaTensor_nDimension(state, input) == 4)
  {
    cudaGradInput  = toDeviceTensor<float, 4>(state, gradInput);
    cudaGradOutput = toDeviceTensor<float, 4>(state, gradOutput);
  }
  else
  {
    cudaGradInput =
      toDeviceTensor<float, 5>(state, gradInput).downcastOuter<4>();
    cudaGradOutput =
      toDeviceTensor<float, 5>(state, gradOutput).downcastOuter<4>();
  }

  THLongStorage *indicesSize = THLongStorage_newWithSize(4);
  long indicesSizeRaw[4] = { batchSize * inputSlices,
                           outputTime, outputHeight, outputWidth };
  THLongStorage_rawCopy(indicesSize, indicesSizeRaw);
  THCIndexTensor *indices1 = THCIndexTensor_(newWithStorage)(
    state, THCIndexTensor_(storage)(state, indices),
    THCIndexTensor_(storageOffset)(state, indices), indicesSize, NULL);
  THLongStorage_free(indicesSize);

  THCDeviceTensor<THCIndex_t, 4> cudaIndices =
    toDeviceTensor<THCIndex_t, 4>(state, indices1);

  int totalZ = outputTime * inputSlices * batchSize;
  int offsetZ = 0;
  dim3 block(32, 8);

  while (totalZ > 0) {
    dim3 grid(THCCeilDiv(outputWidth, static_cast<int>(block.x)),
              THCCeilDiv(outputHeight, static_cast<int>(block.y)),
              totalZ > 65535 ? 65535 : totalZ);

    cuda_VolumetricDilatedMaxPooling_updateGradInput<<<grid, block,
      0, THCState_getCurrentStream(state)>>>(
                                             cudaGradOutput,
                                             cudaIndices,
                                             cudaGradInput,
                                             dT, dH, dW,
                                             padT, padH, padW,
                                             dilationT, dilationH, dilationW, offsetZ);
    THCudaCheck(cudaGetLastError());
    totalZ -= 65535;
    offsetZ += 65535;
  }

  // cleanup
  THCudaTensor_free(state, gradOutput);
  THCIndexTensor_(free)(state, indices1);
}
