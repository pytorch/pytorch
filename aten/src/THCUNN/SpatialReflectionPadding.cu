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

template<typename Dtype>
__global__ void SpatialReflectionPadding_updateOutput(
  THCDeviceTensor<Dtype, 4> input,
  THCDeviceTensor<Dtype, 4> output,
  int padT, int padB, int padL, int padR) {

  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= output.getSize(2) * output.getSize(3)) {
    return;
  }
  int outputPointX = outputPointId % output.getSize(3);
  int outputPointY = outputPointId / output.getSize(3);

  int iStartX = max(0, -padL);
  int iStartY = max(0, -padT);
  int oStartX = max(0, padL);
  int oStartY = max(0, padT);

  int inputPointX = abs(outputPointX - padL)
                  - abs(outputPointX - (input.getSize(3) + padL - 1))
                  - outputPointX
                  + 2 * padL + input.getSize(3) - 1
                  - oStartX + iStartX;

  int inputPointY = abs(outputPointY - padT)
                  - abs(outputPointY - (input.getSize(2) + padT - 1))
                  - outputPointY
                  + 2 * padT + input.getSize(2) - 1
                  - oStartY + iStartY;

  Dtype valueToCopy = input[batch][plane][inputPointY][inputPointX];
  output[batch][plane][outputPointY][outputPointX] = valueToCopy;
}

template <typename Dtype>
__global__ void SpatialReflectionPadding_updateGradInput(
  THCDeviceTensor<Dtype, 4> gradInput,
  THCDeviceTensor<Dtype, 4> gradOutput,
  int padT, int padB, int padL, int padR) {

  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= gradOutput.getSize(2) * gradOutput.getSize(3)) {
    return;
  }
  int outputPointX = outputPointId % gradOutput.getSize(3);
  int outputPointY = outputPointId / gradOutput.getSize(3);

  int iStartX = max(0, -padL);
  int iStartY = max(0, -padT);
  int oStartX = max(0, padL);
  int oStartY = max(0, padT);

  int inputPointX = abs(outputPointX - padL)
                  - abs(outputPointX - (gradInput.getSize(3) + padL - 1))
                  - outputPointX
                  + 2 * padL + gradInput.getSize(3) - 1
                  - oStartX + iStartX;

  int inputPointY = abs(outputPointY - padT)
                  - abs(outputPointY - (gradInput.getSize(2) + padT - 1))
                  - outputPointY
                  + 2 * padT + gradInput.getSize(2) - 1
                  - oStartY + iStartY;

  Dtype valueToCopy = gradOutput[batch][plane][outputPointY][outputPointX];
  atomicAdd(&gradInput[batch][plane][inputPointY][inputPointX], valueToCopy);
}

#include "generic/SpatialReflectionPadding.cu"
#include "THCGenerateFloatTypes.h"
