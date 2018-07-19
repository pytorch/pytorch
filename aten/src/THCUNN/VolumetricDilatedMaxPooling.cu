#include "THCUNN.h"
#include "THCTensor.hpp"
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
  Dtype* inputData, int64_t inputT, int64_t inputH, int64_t inputW,
  THCDeviceTensor<THCIndex_t, 4> indices,
  THCDeviceTensor<Dtype, 4> output,
  int64_t kT, int64_t kH, int64_t kW,
  int64_t dT, int64_t dH, int64_t dW,
  int64_t padT, int64_t padH, int64_t padW,
  int64_t dilationT, int64_t dilationH, int64_t dilationW,
  int64_t offsetZ)
{
  int64_t oColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t oRow    = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t oFrame  = (blockIdx.z + offsetZ) % output.getSize(1); // output frame/time
  int64_t slice   = (blockIdx.z + offsetZ) / output.getSize(1); // output slice/feature

  if (oRow < output.getSize(2) && oColumn < output.getSize(3))
  {
    int64_t tStart = oFrame  * dT - padT;
    int64_t hStart = oRow    * dH - padH;
    int64_t wStart = oColumn * dW - padW;
    int64_t tEnd = fminf(tStart + (kT - 1) * dilationT + 1, inputT);
    int64_t hEnd = fminf(hStart + (kH - 1) * dilationH + 1, inputH);
    int64_t wEnd = fminf(wStart + (kW - 1) * dilationW + 1, inputW);

    while(tStart < 0)
      tStart += dilationT;
    while(hStart < 0)
      hStart += dilationH;
    while(wStart < 0)
      wStart += dilationW;

    int64_t index = 0;
    int64_t maxIndex = -1;
    inputData += slice * inputT * inputH * inputW;

    Dtype max = THCNumerics<Dtype>::min();

    for (int64_t t = tStart; t < tEnd; t += dilationT)
    {
      for (int64_t h = hStart; h < hEnd; h += dilationH)
      {
        for (int64_t w = wStart; w < wEnd; w += dilationW)
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
    indices[slice][oFrame][oRow][oColumn] = maxIndex + TH_INDEX_BASE;
  }
}

template <int64_t KERNEL_WIDTH, typename Dtype>
__global__ void cuda_VolumetricDilatedMaxPooling_updateOutput(
  Dtype* inputData, int64_t inputT, int64_t inputH, int64_t inputW,
  THCDeviceTensor<THCIndex_t, 4> indices,
  THCDeviceTensor<Dtype, 4> output,
  int64_t kT, int64_t kH,
  int64_t dT, int64_t dH, int64_t dW,
  int64_t padT, int64_t padH, int64_t padW,
  int64_t dilationT, int64_t dilationH, int64_t dilationW,
  int64_t offsetZ)
{
  int64_t oColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t oRow    = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t oFrame  = (blockIdx.z + offsetZ) % output.getSize(1); // output frame/time
  int64_t slice   = (blockIdx.z + offsetZ) / output.getSize(1); // output slice/feature

  if (oRow < output.getSize(2) && oColumn < output.getSize(3))
  {
    int64_t tStart = oFrame  * dT - padT;
    int64_t hStart = oRow    * dH - padH;
    int64_t wStart = oColumn * dW - padW;
    int64_t tEnd = fminf(tStart + (kT - 1) * dilationT + 1, inputT);
    int64_t hEnd = fminf(hStart + (kH - 1) * dilationH + 1, inputH);
    int64_t wEnd = fminf(wStart + (KERNEL_WIDTH - 1) * dilationW + 1, inputW);

    while(tStart < 0)
      tStart += dilationT;
    while(hStart < 0)
      hStart += dilationH;
    while(wStart < 0)
      wStart += dilationW;

    int64_t index = 0;
    int64_t maxIndex = -1;

    Dtype max = THCNumerics<Dtype>::min();

    for (int64_t t = tStart; t < tEnd; t += dilationT)
    {
      for (int64_t h = hStart; h < hEnd; h += dilationH)
      {
        for (int64_t w = wStart; w < wEnd; w += dilationW)
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
    indices[slice][oFrame][oRow][oColumn] = maxIndex + TH_INDEX_BASE;
  }
}

template <typename Dtype>
__global__ void cuda_VolumetricDilatedMaxPooling_updateGradInput(
  THCDeviceTensor<Dtype, 4> gradOutput,
  THCDeviceTensor<THCIndex_t, 4> indices,
  Dtype* gradInputData,
  int64_t inputT, int64_t inputH, int64_t inputW,
  int64_t dT, int64_t dH, int64_t dW,
  int64_t padT, int64_t padH, int64_t padW,
  int64_t dilationT, int64_t dilationH, int64_t dilationW,
  int64_t offsetZ)
{
  int64_t oColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t oRow    = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t oFrame  = (blockIdx.z + offsetZ) % gradOutput.getSize(1); // output frame/time
  int64_t slice   = (blockIdx.z + offsetZ) / gradOutput.getSize(1); // output slice/feature

  if (oRow < gradOutput.getSize(2) && oColumn < gradOutput.getSize(3))
  {
    int64_t maxIndex = indices[slice][oFrame][oRow][oColumn] - TH_INDEX_BASE;
    if (maxIndex != -1) {
      atomicAdd(&gradInputData[slice * inputT * inputH * inputW + maxIndex],
                gradOutput[slice][oFrame][oRow][oColumn]);
    }
  }
}

#include "generic/VolumetricDilatedMaxPooling.cu"
#include "THCGenerateFloatTypes.h"
