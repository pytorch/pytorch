#include "THCUNN.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCReduceApplyUtils.cuh"
#include <THC/THCApply.cuh>

#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

template <typename Dtype>
__global__ void TemporalReplicationPadding_updateOutput(
  THCDeviceTensor<Dtype, 3> input,
  THCDeviceTensor<Dtype, 3> output,
  int padL, int padR) {

  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= output.getSize(2)) {
    return;
  }
  int outputPointX = outputPointId % output.getSize(2);

  int iStartX = max(0, -padL);
  int oStartX = max(0, padL);

  int inputPointX = min(max(padL, outputPointX), input.getSize(2) + padL - 1) - oStartX + iStartX;

  Dtype valueToCopy = input[batch][plane][inputPointX];
  output[batch][plane][outputPointX] = valueToCopy;
}

template <typename Dtype>
__global__ void TemporalReplicationPadding_updateGradInput(
  THCDeviceTensor<Dtype, 3> gradInput,
  THCDeviceTensor<Dtype, 3> gradOutput,
  int padL, int padR) {

  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= gradOutput.getSize(2)) {
    return;
  }
  int outputPointX = outputPointId % gradOutput.getSize(2);

  int iStartX = max(0, -padL);
  int oStartX = max(0, padL);

  int inputPointX = min(max(padL, outputPointX), gradInput.getSize(2) + padL - 1) - oStartX + iStartX;

  Dtype valueToCopy = gradOutput[batch][plane][outputPointX];
  atomicAdd(&gradInput[batch][plane][inputPointX], valueToCopy);
}


#include "generic/TemporalReplicationPadding.cu"
#include "THCGenerateFloatTypes.h"
