#include "THCUNN.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include <cfloat>

template <typename Dtype>
__global__ void cuda_VolumetricMaxUnpooling_updateOutput(
  THCDeviceTensor<Dtype, 4> input,
  THCDeviceTensor<THCIndex_t, 4> indices,
  Dtype* outputData,
  int oT, int oH, int oW,
  int dT, int dH, int dW,
  int padT, int padH, int padW, int offsetZ)
{
  int64_t iColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t iRow    = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t iFrame  = (blockIdx.z + offsetZ) % input.getSize(1); // intput frame/time
  int64_t slice   = (blockIdx.z + offsetZ) / input.getSize(1); // intput slice/feature

  if (iRow < input.getSize(2) && iColumn < input.getSize(3))
  {
    Dtype val = input[slice][iFrame][iRow][iColumn];
    int64_t index = indices[slice][iFrame][iRow][iColumn];
    outputData[slice*oT*oH*oW + index] = val;
  }
}

template <typename Dtype>
__global__ void cuda_VolumetricMaxUnpooling_updateGradInput(
  Dtype* gradOutputData,
  int oT, int oH, int oW,
  THCDeviceTensor<THCIndex_t, 4> indices,
  THCDeviceTensor<Dtype, 4> gradInput,
  int dT, int dH, int dW,
  int padT, int padH, int padW, int offsetZ)
{
  int iColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int iRow    = blockIdx.y * blockDim.y + threadIdx.y;
  int iFrame  = (blockIdx.z + offsetZ) % gradInput.getSize(1); // output frame/time
  int slice   = (blockIdx.z + offsetZ) / gradInput.getSize(1); // output slice/feature

  if (iRow < gradInput.getSize(2) && iColumn < gradInput.getSize(3))
  {
    int64_t index = indices[slice][iFrame][iRow][iColumn];
    Dtype grad_val = gradOutputData[slice*oT*oH*oW + index];
    gradInput[slice][iFrame][iRow][iColumn] = grad_val;
  }
}

#include "generic/VolumetricMaxUnpooling.cu"
#include "THCGenerateFloatTypes.h"
