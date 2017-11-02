#include "THCUNN.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

#include <cfloat>

template <typename Dtype>
__global__ void cuda_VolumetricDilatedMaxPooling_updateOutput(
  THCDeviceTensor<Dtype, 4> input,
  THCDeviceTensor<THCIndex_t, 4> indices,
  THCDeviceTensor<Dtype, 4> output,
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

    Dtype max = THCNumerics<Dtype>::min();

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
                Dtype val = input[slice][iFrame + frame * dilationT][iRow + row * dilationH][iColumn + column * dilationW];

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

template <int KERNEL_WIDTH, typename Dtype>
__global__ void cuda_VolumetricDilatedMaxPooling_updateOutput(
  THCDeviceTensor<Dtype, 4> input, THCDeviceTensor<THCIndex_t, 4> indices,
  THCDeviceTensor<Dtype, 4> output,
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

    Dtype max = THCNumerics<Dtype>::min();

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
                Dtype val = input[slice][iFrame + frame * dilationT][iRow + row * dilationH][iColumn + column * dilationW];

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

template <typename Dtype>
__global__ void cuda_VolumetricDilatedMaxPooling_updateGradInput(
  THCDeviceTensor<Dtype, 4> gradOutput,
  THCDeviceTensor<THCIndex_t, 4> indices,
  THCDeviceTensor<Dtype, 4> gradInput,
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

#include "generic/VolumetricDilatedMaxPooling.cu"
#include "THCGenerateFloatTypes.h"
