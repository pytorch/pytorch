#include "THCUNN.h"

#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCReduceApplyUtils.cuh"

__global__ void VolumetricReplicationPadding_updateOutput(
  THCDeviceTensor<float, 5> input,
  THCDeviceTensor<float, 5> output,
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

  float valueToCopy =
      input[batch][plane][inputPointZ][inputPointY][inputPointX];
  output[batch][plane][outputPointZ][outputPointY][outputPointX] = valueToCopy;
}

void THNN_CudaVolumetricReplicationPadding_updateOutput(THCState *state,
                                                        THCudaTensor *input,
                                                        THCudaTensor *output,
                                                        int pleft, int pright,
                                                        int ptop, int pbottom,
                                                        int pfront, int pback) {
  THArgCheck(TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, input), 2,
             "input tensor must fit into 32-bit index math");

  int planeDim = 0;
  int dimd = 1;
  int dimh = 2;
  int dimw = 3;
  int numBatch = 1;

  int numInputDims = THCudaTensor_nDimension(state, input);
  THArgCheck(numInputDims == 4 || numInputDims == 5, 2,
             "input must be 4 or 5-dimensional");

  if (numInputDims == 5) {
    numBatch = THCudaTensor_size(state, input, 0);
    planeDim++;
    dimd++;
    dimh++;
    dimw++;
  }

  int numPlanes = THCudaTensor_size(state, input, planeDim);
  int inputD = THCudaTensor_size(state, input, dimd);
  int inputH = THCudaTensor_size(state, input, dimh);
  int inputW = THCudaTensor_size(state, input, dimw);
  int outputD = inputD + pfront + pback;
  int outputH = inputH + ptop + pbottom;
  int outputW  = inputW + pleft + pright;

  THCDeviceTensor<float, 5> devInput;
  THCDeviceTensor<float, 5> devOutput;

  if (numInputDims == 4) {
    THCudaTensor_resize4d(state, output, numPlanes, outputD, outputH, outputW);

    devInput = toDeviceTensor<float, 4>(state, input).upcastOuter<5>();
    devOutput = toDeviceTensor<float, 4>(state, output).upcastOuter<5>();
  } else {
    THCudaTensor_resize5d(state, output, numBatch, numPlanes, outputD, outputH,
                          outputW);

    devInput = toDeviceTensor<float, 5>(state, input);
    devOutput = toDeviceTensor<float, 5>(state, output);
  }

  int outputPlaneSize = devOutput.getSize(2) * devOutput.getSize(3) *
      devOutput.getSize(4);
  dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devOutput.getSize(1),
            devOutput.getSize(0));
  dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

  VolumetricReplicationPadding_updateOutput<<<gridSize, blockSize, 0, THCState_getCurrentStream(state)>>>(
    devInput, devOutput, pfront, pback, ptop, pbottom, pleft, pright);
}

__global__ void VolumetricReplicationPadding_updateGradInput(
  THCDeviceTensor<float, 5> gradInput,
  THCDeviceTensor<float, 5> gradOutput,
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

  float valueToCopy =
      gradOutput[batch][plane][outputPointZ][outputPointY][outputPointX];
  atomicAdd(&gradInput[batch][plane][inputPointZ][inputPointY][inputPointX],
            valueToCopy);
}

void THNN_CudaVolumetricReplicationPadding_updateGradInput(
  THCState *state, THCudaTensor *input, THCudaTensor *gradOutput,
  THCudaTensor *gradInput, int pleft, int pright, int ptop, int pbottom,
  int pfront, int pback) {
  THArgCheck(TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, input), 2,
             "input tensor must fit into 32-bit index math");
  THArgCheck(TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, gradOutput),
             3, "output gradient tensor must fit into 32-bit index math");

  int planeDim = 0;
  int dimd = 1;
  int dimh = 2;
  int dimw = 3;

  int numInputDims = THCudaTensor_nDimension(state, input);
  if (numInputDims == 5) {
    planeDim++;
    dimd++;
    dimh++;
    dimw++;
  }

  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_zero(state, gradInput);

  THCDeviceTensor<float, 5> devGradInput;
  THCDeviceTensor<float, 5> devGradOutput;

  if (numInputDims == 4) {
    devGradInput = toDeviceTensor<float, 4>(state, gradInput).upcastOuter<5>();
    devGradOutput =
        toDeviceTensor<float, 4>(state, gradOutput).upcastOuter<5>();
  } else {
    devGradInput = toDeviceTensor<float, 5>(state, gradInput);
    devGradOutput = toDeviceTensor<float, 5>(state, gradOutput);
  }

  int outputPlaneSize = devGradOutput.getSize(2) * devGradOutput.getSize(3) *
      devGradOutput.getSize(4);
  dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devGradOutput.getSize(1),
            devGradOutput.getSize(0));
  dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

  VolumetricReplicationPadding_updateGradInput<<<gridSize, blockSize, 0, THCState_getCurrentStream(state)>>>(
    devGradInput, devGradOutput, pfront, pback, ptop, pbottom, pleft, pright);
}
