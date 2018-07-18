#include "THCUNN.h"
#include "THCTensor.hpp"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

template <typename Dtype, typename Acctype>
__global__ void cuda_VolumetricAveragePooling_updateOutput(
  THCDeviceTensor<Dtype, 4> input,
  THCDeviceTensor<Dtype, 4> output,
  int64_t kT, int64_t kH, int64_t kW,
  int64_t dT, int64_t dH, int64_t dW,
  int64_t padT, int64_t padH, int64_t padW,
  bool count_include_pad, int64_t offsetZ)
{
  int64_t oCol   = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t oRow   = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t oFrame = (blockIdx.z + offsetZ) % output.getSize(1); // output frame/time
  int64_t slice  = (blockIdx.z + offsetZ) / output.getSize(1); // output slice/feature

  if (oRow < output.getSize(2) && oCol < output.getSize(3))
  {
    Acctype sum = 0.0;

    int64_t tstart = oFrame * dT - padT;
    int64_t hstart = oRow   * dH - padH;
    int64_t wstart = oCol   * dW - padW;
    int64_t tend = min(tstart + kT, input.getSize(1) + padT);
    int64_t hend = min(hstart + kH, input.getSize(2) + padH);
    int64_t wend = min(wstart + kW, input.getSize(3) + padW);
    int64_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);
    tstart = max(tstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    tend = min(tend, input.getSize(1));
    hend = min(hend, input.getSize(2));
    wend = min(wend, input.getSize(3));

    Acctype divide_factor;
    if (count_include_pad)
      divide_factor = static_cast<Acctype>(pool_size);
    else
      divide_factor = static_cast<Acctype>((tend - tstart) * (hend - hstart) * (wend - wstart));

    int64_t ti, hi, wi;
    for (ti = tstart; ti < tend; ++ti)
    {
      for (hi = hstart; hi < hend; ++hi)
      {
        for (wi = wstart; wi < wend; ++wi)
        {
          Dtype val = input[slice][ti][hi][wi];
          sum += val;
        }
      }
    }

    output[slice][oFrame][oRow][oCol] = ScalarConvert<Acctype, Dtype>::to(sum / divide_factor);
  }
}

// Inner-most loop size (kW) passed as template parameter for
// performance reasons.
//
template<int64_t KERNEL_WIDTH, typename Dtype, typename Acctype>
__global__ void cuda_VolumetricAveragePooling_updateOutput_fixedKW(
  THCDeviceTensor<Dtype, 4> input,
  THCDeviceTensor<Dtype, 4> output,
  int64_t kT, int64_t kH,
  int64_t dT, int64_t dH, int64_t dW,
  int64_t padT, int64_t padH, int64_t padW,
  bool count_include_pad, int64_t offsetZ)
{
  int64_t oCol   = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t oRow   = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t oFrame = (blockIdx.z + offsetZ) % output.getSize(1); // output frame/time
  int64_t slice  = (blockIdx.z + offsetZ) / output.getSize(1); // output slice/feature

  if (oRow < output.getSize(2) && oCol < output.getSize(3))
  {
    Acctype sum = 0.0;

    int64_t tstart = oFrame * dT - padT;
    int64_t hstart = oRow   * dH - padH;
    int64_t wstart = oCol   * dW - padW;
    int64_t tend = min(tstart + kT, input.getSize(1) + padT);
    int64_t hend = min(hstart + kH, input.getSize(2) + padH);
    int64_t wend = min(wstart + KERNEL_WIDTH, input.getSize(3) + padW);
    int64_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);
    tstart = max(tstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    tend = min(tend, input.getSize(1));
    hend = min(hend, input.getSize(2));
    wend = min(wend, input.getSize(3));

    Acctype divide_factor;
    if (count_include_pad)
      divide_factor = static_cast<Acctype>(pool_size);
    else
      divide_factor = static_cast<Acctype>((tend - tstart) * (hend - hstart) * (wend - wstart));

    int64_t ti, hi, wi;
    for (ti = tstart; ti < tend; ++ti)
    {
      for (hi = hstart; hi < hend; ++hi)
      {
        for (wi = wstart; wi < wend; ++wi)
        {
          Dtype val = input[slice][ti][hi][wi];
          sum += val;
        }
      }
    }

    output[slice][oFrame][oRow][oCol] = ScalarConvert<Acctype, Dtype>::to(sum / divide_factor);
  }
}

#define LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(KW) case KW: \
  cuda_VolumetricAveragePooling_updateOutput_fixedKW<KW, real, accreal> \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>( \
      cudaInput, cudaOutput, kT, kH, dT, dH, dW, padT, padH, padW, count_include_pad, offsetZ); \
  break

template <typename Dtype, typename Acctype>
__global__ void cuda_VolumetricAveragePooling_updateGradInput_Stride1(
  THCDeviceTensor<Dtype, 4> gradOutput,
  THCDeviceTensor<Dtype, 4> gradInput,
  int64_t kT, int64_t kH, int64_t kW,
  Acctype normFactor, int64_t offsetZ)
{
  int64_t iCol   = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t iRow   = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t iFrame = (blockIdx.z + offsetZ) % gradInput.getSize(1); // input frame/time
  int64_t slice  = (blockIdx.z + offsetZ) / gradInput.getSize(1); // input slice/feature

  // guard against over-tiled threads
  if (iRow < gradInput.getSize(2) && iCol < gradInput.getSize(3))
  {
    Acctype sum = 0.0;
    Dtype *gOut = &gradOutput[slice][max(0, iFrame - kT + 1)]
      [max(0, iRow - kH + 1)][max(0, iCol - kW + 1)];
    int64_t frameOffset = 0;
    for (int64_t oFrame  = max(0, iFrame - kT + 1);
         oFrame < min(iFrame + 1, gradOutput.getSize(1));
         ++oFrame)
    {
      int64_t rowOffset = frameOffset;
      for (int64_t oRow = max(0, iRow - kH + 1);
           oRow < min(iRow + 1, gradOutput.getSize(2));
           ++oRow)
      {
        int64_t colOffset = rowOffset;
        for (int64_t oCol = max(0, iCol - kW + 1);
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
  int64_t kT, int64_t kH, int64_t kW,
  int64_t dT, int64_t dH, int64_t dW,
  int64_t padT, int64_t padH, int64_t padW,
  bool count_include_pad, int64_t offsetZ)
{
  int64_t oCol   = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t oRow   = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t oFrame = (blockIdx.z + offsetZ) % gradOutput.getSize(1); // gradOutput frame/time
  int64_t slice  = (blockIdx.z + offsetZ) / gradOutput.getSize(1); // gradOutput slice/feature

  // guard against over-tiled threads
  if (oRow < gradOutput.getSize(2) && oCol < gradOutput.getSize(3))
  {
    int64_t tstart = oFrame * dT - padT;
    int64_t hstart = oRow   * dH - padH;
    int64_t wstart = oCol   * dW - padW;
    int64_t tend = min(tstart + kT, gradInput.getSize(1) + padT);
    int64_t hend = min(hstart + kH, gradInput.getSize(2) + padH);
    int64_t wend = min(wstart + kW, gradInput.getSize(3) + padW);
    int64_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);
    tstart = max(tstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    tend = min(tend, gradInput.getSize(1));
    hend = min(hend, gradInput.getSize(2));
    wend = min(wend, gradInput.getSize(3));

    Acctype divide_factor;
    if (count_include_pad)
      divide_factor = static_cast<Acctype>(pool_size);
    else
      divide_factor = static_cast<Acctype>((tend - tstart) * (hend - hstart) * (wend - wstart));

    Dtype val = ScalarConvert<Acctype, Dtype>::to(
      ScalarConvert<Dtype, Acctype>::to(gradOutput[slice][oFrame][oRow][oCol]) / divide_factor);
    for (int64_t iFrame = tstart; iFrame < tend; ++iFrame)
    {
      for (int64_t iRow = hstart; iRow < hend; ++iRow)
      {
        for (int64_t iCol = wstart; iCol < wend; ++iCol)
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
  int64_t kT, int64_t kH, int64_t kW,
  int64_t dT, int64_t dH, int64_t dW,
  int64_t padT, int64_t padH, int64_t padW,
  bool count_include_pad, int64_t offsetZ)
{
  int64_t oCol   = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t oRow   = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t oFrame = (blockIdx.z + offsetZ) % gradOutput.getSize(1); // gradOutput frame/time
  int64_t slice  = (blockIdx.z + offsetZ) / gradOutput.getSize(1); // gradOutput slice/feature

  // guard against over-tiled threads
  if (oRow < gradOutput.getSize(2) && oCol < gradOutput.getSize(3))
  {
    int64_t tstart = oFrame * dT - padT;
    int64_t hstart = oRow   * dH - padH;
    int64_t wstart = oCol   * dW - padW;
    int64_t tend = min(tstart + kT, gradInput.getSize(1) + padT);
    int64_t hend = min(hstart + kH, gradInput.getSize(2) + padH);
    int64_t wend = min(wstart + kW, gradInput.getSize(3) + padW);
    int64_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);
    tstart = max(tstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    tend = min(tend, gradInput.getSize(1));
    hend = min(hend, gradInput.getSize(2));
    wend = min(wend, gradInput.getSize(3));

    Acctype divide_factor;
    if (count_include_pad)
      divide_factor = static_cast<Acctype>(pool_size);
    else
      divide_factor = static_cast<Acctype>((tend - tstart) * (hend - hstart) * (wend - wstart));

    Dtype val = ScalarConvert<Acctype, Dtype>::to(
      ScalarConvert<Dtype, Acctype>::to(gradOutput[slice][oFrame][oRow][oCol]) / divide_factor);
    for (int64_t iFrame = tstart; iFrame < tend; ++iFrame)
    {
      for (int64_t iRow = hstart; iRow < hend; ++iRow)
      {
        for (int64_t iCol = wstart; iCol < wend; ++iCol)
        {
          gradInput[slice][iFrame][iRow][iCol] = val;
        }
      }
    }
  }
}

#include "generic/VolumetricAveragePooling.cu"
#include "THCGenerateFloatTypes.h"
