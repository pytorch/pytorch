#include "THCUNN.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

template <typename Dtype, typename Acctype>
__global__ void cuda_VolumetricAveragePooling_updateOutput(
  THCDeviceTensor<Dtype, 4> input, THCDeviceTensor<Dtype, 4> output,
  int kT, int kH, int kW, int dT, int dH, int dW, Acctype normFactor, int offsetZ)
{
  int oCol   = blockIdx.x * blockDim.x + threadIdx.x;
  int oRow   = blockIdx.y * blockDim.y + threadIdx.y;
  int oFrame = (blockIdx.z + offsetZ) % output.getSize(1); // output frame/time
  int slice  = (blockIdx.z + offsetZ) / output.getSize(1); // output slice/feature

  if (oRow < output.getSize(2) && oCol < output.getSize(3))
  {
    Acctype sum = 0.0;

    int iColumn = oCol * dW;
    int iRow    = oRow    * dH;
    int iFrame  = oFrame  * dT;

    for (int frame = 0; frame < kT; ++frame)
    {
      if (iFrame + frame < input.getSize(1))
      {
        for (int row = 0; row < kH; ++row)
        {
          if (iRow + row < input.getSize(2))
          {
            for (int column = 0; column < kW; ++column)
            {
              if (iColumn + column < input.getSize(3))
              {
                Dtype val = input[slice][iFrame + frame][iRow + row][iColumn + column];
                sum += val;
              }
            }
          }
        }
      }
    }

    output[slice][oFrame][oRow][oCol] = ScalarConvert<Acctype, Dtype>::to(sum * normFactor);
  }
}

// Inner-most loop size (kW) passed as template parameter for
// performance reasons.
//
template<int KERNEL_WIDTH, typename Dtype, typename Acctype>
__global__ void cuda_VolumetricAveragePooling_updateOutput(
  THCDeviceTensor<Dtype, 4> input, THCDeviceTensor<Dtype, 4> output,
  int kT, int kH, int dT, int dH, int dW, Acctype normFactor, int offsetZ)
{
  int oCol   = blockIdx.x * blockDim.x + threadIdx.x;
  int oRow   = blockIdx.y * blockDim.y + threadIdx.y;
  int oFrame = (blockIdx.z + offsetZ) % output.getSize(1); // output frame/time
  int slice  = (blockIdx.z + offsetZ) / output.getSize(1); // output slice/feature

  if (oRow < output.getSize(2) && oCol < output.getSize(3))
  {
    Acctype sum = 0.0;

    int iColumn = oCol * dW;
    int iRow    = oRow    * dH;
    int iFrame  = oFrame  * dT;

    for (int frame = 0; frame < kT; ++frame)
    {
      if (iFrame + frame < input.getSize(1))
      {
        for (int row = 0; row < kH; ++row)
        {
          if (iRow + row < input.getSize(2))
          {
            for (int column = 0; column < KERNEL_WIDTH; ++column)
            {
              if (iColumn + column < input.getSize(3))
              {
                Dtype val = input[slice][iFrame + frame][iRow + row][iColumn + column];
                sum += val;
              }
            }
          }
        }
      }
    }

    output[slice][oFrame][oRow][oCol] = ScalarConvert<Acctype, Dtype>::to(sum * normFactor);
  }
}

#define LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(KW) case KW:                  \
  cuda_VolumetricAveragePooling_updateOutput<KW><<<grid, block>>>(      \
    cudaInput, cudaOutput, kT, kH, dT, dH, dW, normFactor, offsetZ); \
  break

template <typename Dtype, typename Acctype>
__global__ void cuda_VolumetricAveragePooling_updateGradInput_Stride1(
  THCDeviceTensor<Dtype, 4> gradOutput,
  THCDeviceTensor<Dtype, 4> gradInput,
  int kT, int kH, int kW, Acctype normFactor, int offsetZ)
{
  int iCol   = blockIdx.x * blockDim.x + threadIdx.x;
  int iRow   = blockIdx.y * blockDim.y + threadIdx.y;
  int iFrame = (blockIdx.z + offsetZ) % gradInput.getSize(1); // input frame/time
  int slice  = (blockIdx.z + offsetZ) / gradInput.getSize(1); // input slice/feature

  // guard against over-tiled threads
  if (iRow < gradInput.getSize(2) && iCol < gradInput.getSize(3))
  {
    Acctype sum = 0.0;
    Dtype *gOut = &gradOutput[slice][max(0, iFrame - kT + 1)]
      [max(0, iRow - kH + 1)][max(0, iCol - kW + 1)];
    int frameOffset = 0;
    for (int oFrame  = max(0, iFrame - kT + 1);
         oFrame < min(iFrame + 1, gradOutput.getSize(1));
         ++oFrame)
    {
      int rowOffset = frameOffset;
      for (int oRow = max(0, iRow - kH + 1);
           oRow < min(iRow + 1, gradOutput.getSize(2));
           ++oRow)
      {
        int colOffset = rowOffset;
        for (int oCol = max(0, iCol - kW + 1);
             oCol < min(iCol + 1, gradOutput.getSize(3));
             ++oCol)
        {
          sum += gOut[colOffset];
          ++colOffset;
        }
        rowOffset += gradOutput.getSize(3);
      }
      frameOffset += gradOutput.getSize(2) * gradOutput.getSize(3);
    }
    gradInput[slice][iFrame][iRow][iCol] = ScalarConvert<Acctype, Dtype>::to(sum * normFactor);
  }
}

template <typename Dtype, typename Acctype>
__global__ void cuda_VolumetricAveragePooling_updateGradInput_atomicAdd(
  THCDeviceTensor<Dtype, 4> gradOutput,
  THCDeviceTensor<Dtype, 4> gradInput,
  int kT, int kH, int kW, int dT, int dH, int dW, int offsetZ)
{
  int oCol   = blockIdx.x * blockDim.x + threadIdx.x;
  int oRow   = blockIdx.y * blockDim.y + threadIdx.y;
  int oFrame = (blockIdx.z + offsetZ) % gradOutput.getSize(1); // gradOutput frame/time
  int slice  = (blockIdx.z + offsetZ) / gradOutput.getSize(1); // gradOutput slice/feature

  // guard against over-tiled threads
  if (oRow < gradOutput.getSize(2) && oCol < gradOutput.getSize(3))
  {
    Dtype val = ScalarConvert<Acctype, Dtype>::to(
      ScalarConvert<Dtype, Acctype>::to(gradOutput[slice][oFrame][oRow][oCol]) / (kT * kH * kW));
    for (int iFrame = oFrame * dT; iFrame < oFrame * dT + kT; ++iFrame)
    {
      for (int iRow = oRow * dH; iRow < oRow * dH + kH; ++iRow)
      {
        for (int iCol = oCol * dW; iCol < oCol * dW + kW; ++iCol)
        {
          atomicAdd(&gradInput[slice][iFrame][iRow][iCol], val);
        }
      }
    }
  }
}

template <typename Dtype, typename Acctype>
__global__ void cuda_VolumetricAveragePooling_updateGradInput(
  THCDeviceTensor<Dtype, 4> gradOutput,
  THCDeviceTensor<Dtype, 4> gradInput,
  int kT, int kH, int kW,
  int dT, int dH, int dW, int offsetZ)
{
  int oCol   = blockIdx.x * blockDim.x + threadIdx.x;
  int oRow   = blockIdx.y * blockDim.y + threadIdx.y;
  int oFrame = (blockIdx.z + offsetZ) % gradOutput.getSize(1); // gradOutput frame/time
  int slice  = (blockIdx.z + offsetZ) / gradOutput.getSize(1); // gradOutput slice/feature

  // guard against over-tiled threads
  if (oRow < gradOutput.getSize(2) && oCol < gradOutput.getSize(3))
  {
    Dtype val = ScalarConvert<Acctype, Dtype>::to(
      ScalarConvert<Dtype, Acctype>::to(gradOutput[slice][oFrame][oRow][oCol]) / (kT * kH * kW));
    for (int iFrame = oFrame * dT; iFrame < oFrame * dT + kT; ++iFrame)
    {
      for (int iRow = oRow * dH; iRow < oRow * dH + kH; ++iRow)
      {
        for (int iCol = oCol * dW; iCol < oCol * dW + kW; ++iCol)
        {
          gradInput[slice][iFrame][iRow][iCol] = val;
        }
      }
    }
  }
}

#include "generic/VolumetricAveragePooling.cu"
#include "THCGenerateFloatTypes.h"
