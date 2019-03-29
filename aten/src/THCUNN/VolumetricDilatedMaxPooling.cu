#include <THCUNN/THCUNN.h>
#include <THC/THCTensor.hpp>
#include <THCUNN/common.h>
#include <THC/THCDeviceTensor.cuh>
#include <THC/THCDeviceTensorUtils.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCAtomics.cuh>

template <typename Dtype>
__global__ void cuda_VolumetricDilatedMaxPooling_updateOutput(
  Dtype* inputData, int inputT, int inputH, int inputW,
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
    int tStart = oFrame  * dT - padT;
    int hStart = oRow    * dH - padH;
    int wStart = oColumn * dW - padW;
    int tEnd = min(tStart + (kT - 1) * dilationT + 1, inputT);
    int hEnd = min(hStart + (kH - 1) * dilationH + 1, inputH);
    int wEnd = min(wStart + (kW - 1) * dilationW + 1, inputW);

    while(tStart < 0)
      tStart += dilationT;
    while(hStart < 0)
      hStart += dilationH;
    while(wStart < 0)
      wStart += dilationW;

    int index = 0;
    int maxIndex = -1;
    inputData += slice * inputT * inputH * inputW;

    Dtype max = THCNumerics<Dtype>::min();

    for (int t = tStart; t < tEnd; t += dilationT)
    {
      for (int h = hStart; h < hEnd; h += dilationH)
      {
        for (int w = wStart; w < wEnd; w += dilationW)
        {
          index = t * inputH * inputW + h * inputW + w;
          Dtype val = inputData[index];

          if ((max < val) || THCNumerics<Dtype>::isnan(val))
          {
            max = val;
            maxIndex = index;
          }
        }
      }
    }

    output[slice][oFrame][oRow][oColumn] = max;
    indices[slice][oFrame][oRow][oColumn] = maxIndex;
  }
}

template <int KERNEL_WIDTH, typename Dtype>
__global__ void cuda_VolumetricDilatedMaxPooling_updateOutput(
  Dtype* inputData, int inputT, int inputH, int inputW,
  THCDeviceTensor<THCIndex_t, 4> indices,
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
    int tStart = oFrame  * dT - padT;
    int hStart = oRow    * dH - padH;
    int wStart = oColumn * dW - padW;
    int tEnd = min(tStart + (kT - 1) * dilationT + 1, inputT);
    int hEnd = min(hStart + (kH - 1) * dilationH + 1, inputH);
    int wEnd = min(wStart + (KERNEL_WIDTH - 1) * dilationW + 1, inputW);

    while(tStart < 0)
      tStart += dilationT;
    while(hStart < 0)
      hStart += dilationH;
    while(wStart < 0)
      wStart += dilationW;

    int index = 0;
    int maxIndex = -1;

    Dtype max = THCNumerics<Dtype>::min();

    for (int t = tStart; t < tEnd; t += dilationT)
    {
      for (int h = hStart; h < hEnd; h += dilationH)
      {
        for (int w = wStart; w < wEnd; w += dilationW)
        {
          index = t * inputH * inputW + h * inputW + w;
          Dtype val = inputData[slice * inputT * inputH * inputW + index];

          if (max < val)
          {
            max = val;
            maxIndex = index;
          }
        }
      }
    }

    output[slice][oFrame][oRow][oColumn] = max;
    indices[slice][oFrame][oRow][oColumn] = maxIndex;
  }
}

template <typename Dtype>
__global__ void cuda_VolumetricDilatedMaxPooling_updateGradInput(
  THCDeviceTensor<Dtype, 4> gradOutput,
  THCDeviceTensor<THCIndex_t, 4> indices,
  Dtype* gradInputData,
  int inputT, int inputH, int inputW,
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
    int maxIndex = indices[slice][oFrame][oRow][oColumn];
    if (maxIndex != -1) {
      atomicAdd(&gradInputData[slice * inputT * inputH * inputW + maxIndex],
                gradOutput[slice][oFrame][oRow][oColumn]);
    }
  }
}

#include <THCUNN/generic/VolumetricDilatedMaxPooling.cu>
#include <THC/THCGenerateFloatTypes.h>
