#include "THCUNN.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"

#include <cfloat>

__device__ inline float getInterval(float sample,
                                    int index,
                                    int inputSize,
                                    int outputSize,
                                    int poolSize) {
  float alpha = (float)(inputSize - poolSize) / (float) (outputSize - 1);
  if (index == outputSize - 1) {
    return inputSize - poolSize;
  } else {
    return (int) ((index + sample) * alpha) - (int) (sample * alpha);
  }
}

// We template on poolSizeW to allow the innermost loop to be unrolled
template <int PoolSizeWStatic>
__global__ void SpatialFractionalMaxPooling_updateOutput(
  THCDeviceTensor<float, 4> input,
  THCDeviceTensor<float, 4> output,
  THCDeviceTensor<float, 4> indices,
  THCDeviceTensor<float, 3> samples,
  int poolSizeW, int poolSizeH) {

  // Output (h, w) point that this thread is responsible for
  int ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;

  // Each thread generates a specific output point
  if (ourOutputPoint < output.getSize(2) * output.getSize(3)) {
    int outputW = ourOutputPoint % output.getSize(3);
    int outputH = ourOutputPoint / output.getSize(3);

    int poolW = getInterval(samples[batch][plane][0], outputW,
                            input.getSize(3), output.getSize(3), poolSizeW);
    int poolH = getInterval(samples[batch][plane][1], outputH,
                            input.getSize(2), output.getSize(2), poolSizeH);

    float maxVal = -FLT_MAX;
    int maxIndex = -1;

    for (int h = poolH; h < poolH + poolSizeH; ++h) {
      if (PoolSizeWStatic == -1) {
        for (int w = poolW; w < poolW + poolSizeW; ++w) {
          float val = input[batch][plane][h][w];
          maxVal = fmaxf(val, maxVal);
          maxIndex = (maxVal == val) ? (h * input.getSize(3) + w) : maxIndex;
        }
      } else {
#pragma unroll
        for (int i = 0; i < PoolSizeWStatic; ++i) {
          int w = i + poolW;
          float val = input[batch][plane][h][w];
          maxVal = fmaxf(val, maxVal);
          maxIndex = (maxVal == val) ? (h * input.getSize(3) + w) : maxIndex;
        }
      }
    }

    assert(maxVal != -FLT_MAX);
    assert(maxIndex != -1);

    // +1 for Lua index
    indices[batch][plane][outputH][outputW] = maxIndex + 1;
    output[batch][plane][outputH][outputW] = maxVal;
  }
}

void THNN_CudaSpatialFractionalMaxPooling_updateOutput(
    THCState *state,
    THCudaTensor *input,
    THCudaTensor *output,
    int outputW, int outputH,
    int poolSizeW, int poolSizeH,
    THCudaTensor *indices,
    THCudaTensor *randomSamples)
{
  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;
  long numBatch = 1;

  long numInputDims = THCudaTensor_nDimension(state, input);
  THArgCheck(numInputDims == 3 || numInputDims == 4, 2,
                "3D or 4D (batch mode) tensor expected");

  if (numInputDims == 4) {
    numBatch = THCudaTensor_size(state, input, 0);
    planeDim++;
    dimh++;
    dimw++;
  }

  /* sizes */
  long numPlanes = THCudaTensor_size(state, input, planeDim);
  long inputH = THCudaTensor_size(state, input, dimh);
  long inputW = THCudaTensor_size(state, input, dimw);

  THArgCheck(outputH + poolSizeH - 1 < inputH, 6,
                "poolSizeH too large relative to input height");
  THArgCheck(outputW + poolSizeW - 1 < inputW, 5,
                "poolSizeW too large relative to input width");

  THCDeviceTensor<float, 4> devInput;
  THCDeviceTensor<float, 4> devOutput;
  THCDeviceTensor<float, 4> devIndices;
  THCDeviceTensor<float, 3> devSamples =
    toDeviceTensor<float, 3>(state, randomSamples);

  if (numInputDims == 3) {
    /* resize output */
    THCudaTensor_resize3d(state, output, numPlanes, outputH, outputW);
    /* indices will contain the locations for each output point */
    THCudaTensor_resize3d(state, indices, numPlanes, outputH, outputW);

    devInput = toDeviceTensor<float, 3>(state, input).upcastOuter<4>();
    devOutput = toDeviceTensor<float, 3>(state, output).upcastOuter<4>();
    devIndices = toDeviceTensor<float, 3>(state, indices).upcastOuter<4>();
  } else {
    THCudaTensor_resize4d(state, output, numBatch, numPlanes, outputH, outputW);
    /* indices will contain the locations for each output point */
    THCudaTensor_resize4d(state, indices, numBatch, numPlanes, outputH, outputW);

    devInput = toDeviceTensor<float, 4>(state, input);
    devOutput = toDeviceTensor<float, 4>(state, output);
    devIndices = toDeviceTensor<float, 4>(state, indices);
  }

  // block is limited to 4 warps
  // grid handles overflow per each plane
  int outputPlaneSize = devOutput.getSize(2) * devOutput.getSize(3);
  dim3 grid(THCCeilDiv(outputPlaneSize, 128),
            devInput.getSize(1),
            devInput.getSize(0));
  dim3 block(outputPlaneSize > 128 ? 128 : outputPlaneSize);

#define SFMP_UPDATE_OUTPUT(POOL_W)                                      \
  SpatialFractionalMaxPooling_updateOutput<POOL_W>                      \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(             \
      devInput, devOutput, devIndices, devSamples, poolSizeW, poolSizeH);

#define SFMP_UPDATE_OUTPUT_CASE(POOL_W)                 \
  case POOL_W: SFMP_UPDATE_OUTPUT(POOL_W); break

  switch (poolSizeW) {
    SFMP_UPDATE_OUTPUT_CASE(2);
    SFMP_UPDATE_OUTPUT_CASE(3);
    SFMP_UPDATE_OUTPUT_CASE(4);
    SFMP_UPDATE_OUTPUT_CASE(5);
    SFMP_UPDATE_OUTPUT_CASE(6);
    SFMP_UPDATE_OUTPUT_CASE(7);
    default:
      // dynamic pool width
      SFMP_UPDATE_OUTPUT_CASE(-1);
  }
}

__global__ void SpatialFractionalMaxPooling_updateGradInput(
  THCDeviceTensor<float, 4> gradInput,
  THCDeviceTensor<float, 4> gradOutput,
  THCDeviceTensor<float, 4> indices) {
  // Output (h, w) point that this thread is responsible for
  int ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;

  // Each thread generates a specific output point
  if (ourOutputPoint < gradOutput.getSize(2) * gradOutput.getSize(3)) {
    int outputW = ourOutputPoint % gradOutput.getSize(3);
    int outputH = ourOutputPoint / gradOutput.getSize(3);

    int index = indices[batch][plane][outputH][outputW] - 1;
    assert(index >= 0);
    int inputW = index % gradInput.getSize(3);
    int inputH = index / gradInput.getSize(3);
    assert(inputH < gradInput.getSize(2));

    atomicAdd(gradInput[batch][plane][inputH][inputW].data(),
              gradOutput[batch][plane][outputH][outputW]);
  }
}

void THNN_CudaSpatialFractionalMaxPooling_updateGradInput(
    THCState *state,
    THCudaTensor *input,
    THCudaTensor *gradOutput,
    THCudaTensor *gradInput,
    int outputW, int outputH,
    int poolSizeW, int poolSizeH,
    THCudaTensor *indices)
{
  int dimh = 1;
  int dimw = 2;

  long numInputDims = THCudaTensor_nDimension(state, input);
  if (numInputDims == 4) {
    dimh++;
    dimw++;
  }

  /* sizes */
  long inputH = THCudaTensor_size(state, input, dimh);
  long inputW = THCudaTensor_size(state, input, dimw);

  THArgCheck(outputH == THCudaTensor_size(state, gradOutput, dimh), 3,
                "gradOutput height unexpected");
  THArgCheck(outputW == THCudaTensor_size(state, gradOutput, dimw), 3,
                "gradOutput width unexpected");

  /* resize */
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_zero(state, gradInput);

  THCDeviceTensor<float, 4> devGradInput;
  THCDeviceTensor<float, 4> devGradOutput;
  THCDeviceTensor<float, 4> devIndices;

  /* backprop */
  if (numInputDims == 3) {
    devGradInput = toDeviceTensor<float, 3>(state, gradInput).upcastOuter<4>();
    devGradOutput = toDeviceTensor<float, 3>(state, gradOutput).upcastOuter<4>();
    devIndices = toDeviceTensor<float, 3>(state, indices).upcastOuter<4>();
  } else {
    devGradInput = toDeviceTensor<float, 4>(state, gradInput);
    devGradOutput = toDeviceTensor<float, 4>(state, gradOutput);
    devIndices = toDeviceTensor<float, 4>(state, indices);
  }

  // block is limited to 4 warps
  // grid handles overflow per each plane
  int outputPlaneSize = devGradOutput.getSize(2) * devGradOutput.getSize(3);
  dim3 grid(THCCeilDiv(outputPlaneSize, 128),
            devGradInput.getSize(1),
            devGradInput.getSize(0));
  dim3 block(outputPlaneSize > 128 ? 128 : outputPlaneSize);

  SpatialFractionalMaxPooling_updateGradInput
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      devGradInput, devGradOutput, devIndices);
}
