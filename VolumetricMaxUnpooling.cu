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
  THCDeviceTensor<Dtype, 4> output,
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

    Dtype val = input[slice][iFrame][iRow][iColumn];

    THCIndex_t *idx = &indices[slice][iFrame][iRow][iColumn];
    long maxz = ((unsigned char*)(idx))[0];
    long maxy = ((unsigned char*)(idx))[1];
    long maxx = ((unsigned char*)(idx))[2];
    output[slice][start_t + maxz][start_h + maxy][start_w + maxx] = val;
  }
}

template <typename Dtype>
__global__ void cuda_VolumetricMaxUnpooling_updateGradInput(
  THCDeviceTensor<Dtype, 4> gradOutput,
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

    long start_t = iFrame * dT - padT;
    long start_h = iRow * dH - padH;
    long start_w = iColumn * dW - padW;

    THCIndex_t *idx = &indices[slice][iFrame][iRow][iColumn];
    long maxz = ((unsigned char*)(idx))[0];
    long maxy = ((unsigned char*)(idx))[1];
    long maxx = ((unsigned char*)(idx))[2];

    Dtype grad_val = gradOutput[slice][start_t + maxz][start_h + maxy][start_w + maxx];

    gradInput[slice][iFrame][iRow][iColumn] = grad_val;
  }
}

#include "generic/VolumetricMaxUnpooling.cu"
#include "THCGenerateFloatTypes.h"
