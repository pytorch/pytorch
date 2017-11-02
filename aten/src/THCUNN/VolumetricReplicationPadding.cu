#include "THCUNN.h"
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
  int pfront, int pback, int ptop, int pbottom, int pleft, int pright) {

  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= (output.getSize(2) * output.getSize(3) *
                        output.getSize(4))) {
    return;
  }
  int outputPointX = outputPointId % output.getSize(4);
  int outputPointY = (outputPointId / output.getSize(4)) % output.getSize(3);
  int outputPointZ = outputPointId / (output.getSize(3) * output.getSize(4));

  int iStartX = max(0, -pleft);
  int iStartY = max(0, -ptop);
  int iStartZ = max(0, -pfront);
  int oStartX = max(0, pleft);
  int oStartY = max(0, ptop);
  int oStartZ = max(0, pfront);

  int inputPointX = min(max(pleft, outputPointX),
                        input.getSize(4) + pleft - 1) - oStartX + iStartX;
  int inputPointY = min(max(ptop, outputPointY),
                        input.getSize(3) + ptop - 1) - oStartY + iStartY;
  int inputPointZ = min(max(pfront, outputPointZ),
                        input.getSize(2) + pfront - 1) - oStartZ + iStartZ;

  Dtype valueToCopy =
      input[batch][plane][inputPointZ][inputPointY][inputPointX];
  output[batch][plane][outputPointZ][outputPointY][outputPointX] = valueToCopy;
}

template <typename Dtype>
__global__ void VolumetricReplicationPadding_updateGradInput(
  THCDeviceTensor<Dtype, 5> gradInput,
  THCDeviceTensor<Dtype, 5> gradOutput,
  int pfront, int pback, int ptop, int pbottom, int pleft, int pright) {
  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;

  if (outputPointId >= (gradOutput.getSize(2) * gradOutput.getSize(3) *
                        gradOutput.getSize(4))) {
    return;
  }
  int outputPointX = outputPointId % gradOutput.getSize(4);
  int outputPointY = (outputPointId / gradOutput.getSize(4)) %
      gradOutput.getSize(3);
  int outputPointZ = outputPointId / (gradOutput.getSize(3) *
      gradOutput.getSize(4));

  int iStartX = max(0, -pleft);
  int iStartY = max(0, -ptop);
  int iStartZ = max(0, -pfront);
  int oStartX = max(0, pleft);
  int oStartY = max(0, ptop);
  int oStartZ = max(0, pfront);

  int inputPointX = min(max(pleft, outputPointX),
                        gradInput.getSize(4) + pleft - 1) - oStartX + iStartX;
  int inputPointY = min(max(ptop, outputPointY),
                        gradInput.getSize(3) + ptop - 1) - oStartY + iStartY;
  int inputPointZ = min(max(pfront, outputPointZ),
                        gradInput.getSize(2) + pfront - 1) - oStartZ + iStartZ;

  Dtype valueToCopy =
      gradOutput[batch][plane][outputPointZ][outputPointY][outputPointX];
  atomicAdd(&gradInput[batch][plane][inputPointZ][inputPointY][inputPointX],
            valueToCopy);
}


#include "generic/VolumetricReplicationPadding.cu"
#include "THCGenerateFloatTypes.h"
