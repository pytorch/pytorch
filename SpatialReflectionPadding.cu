#include "THCUNN.h"

#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCReduceApplyUtils.cuh"

__global__ void SpatialReflectionPadding_updateOutput(
  THCDeviceTensor<float, 4> input,
  THCDeviceTensor<float, 4> output,
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

  float valueToCopy = input[batch][plane][inputPointY][inputPointX];
  output[batch][plane][outputPointY][outputPointX] = valueToCopy;
}

void THNN_CudaSpatialReflectionPadding_updateOutput(THCState *state,
                                                    THCudaTensor *input,
                                                    THCudaTensor *output,
                                                    int padL, int padR,
                                                    int padT, int padB
                                                   ) {
  THArgCheck(TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, input), 2,
             "input tensor must fit into 32-bit index math");

  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;
  int numBatch = 1;

  int numInputDims = THCudaTensor_nDimension(state, input);
  THArgCheck(numInputDims == 3 || numInputDims == 4, 2,
                "input must be 3 or 4-dimensional");

  if (numInputDims == 4) {
    numBatch = THCudaTensor_size(state, input, 0);
    planeDim++;
    dimh++;
    dimw++;
  }

  int numPlanes = THCudaTensor_size(state, input, planeDim);
  int inputH = THCudaTensor_size(state, input, dimh);
  int inputW = THCudaTensor_size(state, input, dimw);
  int outputH = inputH + padT + padB;
  int outputW  = inputW + padL + padR;

  THCDeviceTensor<float, 4> devInput;
  THCDeviceTensor<float, 4> devOutput;

  if (numInputDims == 3) {
    THCudaTensor_resize3d(state, output, numPlanes, outputH, outputW);

    devInput = toDeviceTensor<float, 3>(state, input).upcastOuter<4>();
    devOutput = toDeviceTensor<float, 3>(state, output).upcastOuter<4>();
  } else {
    THCudaTensor_resize4d(state, output, numBatch, numPlanes, outputH, outputW);

    devInput = toDeviceTensor<float, 4>(state, input);
    devOutput = toDeviceTensor<float, 4>(state, output);
  }

  int outputPlaneSize = devOutput.getSize(2) * devOutput.getSize(3);
  dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devOutput.getSize(1),
            devOutput.getSize(0));
  dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

  SpatialReflectionPadding_updateOutput<<<gridSize, blockSize, 0, THCState_getCurrentStream(state)>>>(
    devInput, devOutput, padT, padB, padL, padR);
  THCudaCheck(cudaGetLastError());
}

__global__ void SpatialReflectionPadding_updateGradInput(
  THCDeviceTensor<float, 4> gradInput,
  THCDeviceTensor<float, 4> gradOutput,
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

  float valueToCopy = gradOutput[batch][plane][outputPointY][outputPointX];
  atomicAdd(&gradInput[batch][plane][inputPointY][inputPointX], valueToCopy);
}

void THNN_CudaSpatialReflectionPadding_updateGradInput(THCState *state,
                                                       THCudaTensor *input,
                                                       THCudaTensor *gradOutput,
                                                       THCudaTensor *gradInput,
                                                       int padL, int padR,
                                                       int padT, int padB) {

  THArgCheck(TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, input), 2,
                "input tensor must fit into 32-bit index math");
  THArgCheck(TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, gradOutput), 3,
                "output gradient tensor must fit into 32-bit index math");

  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;

  int numInputDims = THCudaTensor_nDimension(state, input);
  if (numInputDims == 4) {
    planeDim++;
    dimh++;
    dimw++;
  }

  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_zero(state, gradInput);

  THCDeviceTensor<float, 4> devGradInput;
  THCDeviceTensor<float, 4> devGradOutput;

  if (numInputDims == 3) {
    devGradInput = toDeviceTensor<float, 3>(state, gradInput).upcastOuter<4>();
    devGradOutput = toDeviceTensor<float, 3>(state, gradOutput).upcastOuter<4>();
  } else {
    devGradInput = toDeviceTensor<float, 4>(state, gradInput);
    devGradOutput = toDeviceTensor<float, 4>(state, gradOutput);
  }

  int outputPlaneSize = devGradOutput.getSize(2) * devGradOutput.getSize(3);
  dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devGradOutput.getSize(1),
            devGradOutput.getSize(0));
  dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

  SpatialReflectionPadding_updateGradInput<<<gridSize, blockSize, 0, THCState_getCurrentStream(state)>>>(
    devGradInput, devGradOutput, padT, padB, padL, padR);
  THCudaCheck(cudaGetLastError());
}
