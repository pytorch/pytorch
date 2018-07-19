#include "THCUNN.h"
#include "THCTensor.hpp"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCReduceApplyUtils.cuh"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"
#include <THC/THCApply.cuh>

template <typename Dtype>
__global__ void VolumetricReplicationPadding_updateOutput(
  THCDeviceTensor<Dtype, 5> input,
  THCDeviceTensor<Dtype, 5> output,
  int64_t pfront, int64_t pback, int64_t ptop, int64_t pbottom, int64_t pleft, int64_t pright) {

  int64_t outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t plane = blockIdx.y;
  int64_t batch = blockIdx.z;
  if (outputPointId >= (output.getSize(2) * output.getSize(3) *
                        output.getSize(4))) {
    return;
  }
  int64_t outputPointX = outputPointId % output.getSize(4);
  int64_t outputPointY = (outputPointId / output.getSize(4)) % output.getSize(3);
  int64_t outputPointZ = outputPointId / (output.getSize(3) * output.getSize(4));

  int64_t iStartX = max(0, -pleft);
  int64_t iStartY = max(0, -ptop);
  int64_t iStartZ = max(0, -pfront);
  int64_t oStartX = max(0, pleft);
  int64_t oStartY = max(0, ptop);
  int64_t oStartZ = max(0, pfront);

  int64_t inputPointX = min(max(pleft, outputPointX),
                        input.getSize(4) + pleft - 1) - oStartX + iStartX;
  int64_t inputPointY = min(max(ptop, outputPointY),
                        input.getSize(3) + ptop - 1) - oStartY + iStartY;
  int64_t inputPointZ = min(max(pfront, outputPointZ),
                        input.getSize(2) + pfront - 1) - oStartZ + iStartZ;

  Dtype valueToCopy =
      input[batch][plane][inputPointZ][inputPointY][inputPointX];
  output[batch][plane][outputPointZ][outputPointY][outputPointX] = valueToCopy;
}

template <typename Dtype>
__global__ void VolumetricReplicationPadding_updateGradInput(
  THCDeviceTensor<Dtype, 5> gradInput,
  THCDeviceTensor<Dtype, 5> gradOutput,
  int64_t pfront, int64_t pback, int64_t ptop, int64_t pbottom, int64_t pleft, int64_t pright) {
  int64_t outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t plane = blockIdx.y;
  int64_t batch = blockIdx.z;

  if (outputPointId >= (gradOutput.getSize(2) * gradOutput.getSize(3) *
                        gradOutput.getSize(4))) {
    return;
  }
  int64_t outputPointX = outputPointId % gradOutput.getSize(4);
  int64_t outputPointY = (outputPointId / gradOutput.getSize(4)) %
      gradOutput.getSize(3);
  int64_t outputPointZ = outputPointId / (gradOutput.getSize(3) *
      gradOutput.getSize(4));

  int64_t iStartX = max(0, -pleft);
  int64_t iStartY = max(0, -ptop);
  int64_t iStartZ = max(0, -pfront);
  int64_t oStartX = max(0, pleft);
  int64_t oStartY = max(0, ptop);
  int64_t oStartZ = max(0, pfront);

  int64_t inputPointX = min(max(pleft, outputPointX),
                        gradInput.getSize(4) + pleft - 1) - oStartX + iStartX;
  int64_t inputPointY = min(max(ptop, outputPointY),
                        gradInput.getSize(3) + ptop - 1) - oStartY + iStartY;
  int64_t inputPointZ = min(max(pfront, outputPointZ),
                        gradInput.getSize(2) + pfront - 1) - oStartZ + iStartZ;

  Dtype valueToCopy =
      gradOutput[batch][plane][outputPointZ][outputPointY][outputPointX];
  atomicAdd(&gradInput[batch][plane][inputPointZ][inputPointY][inputPointX],
            valueToCopy);
}


#include "generic/VolumetricReplicationPadding.cu"
#include "THCGenerateFloatTypes.h"
