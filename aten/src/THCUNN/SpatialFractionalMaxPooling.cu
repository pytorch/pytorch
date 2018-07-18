#include "THCUNN.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

#include <cfloat>

template <typename Dtype, typename Acctype>
__device__ inline int64_t getInterval(Acctype sample,
                                  int64_t index,
                                  int64_t inputSize,
                                  int64_t outputSize,
                                  int64_t poolSize) {
  Acctype alpha = (Acctype)(inputSize - poolSize) / (Acctype) (outputSize - 1);
  if (index == outputSize - 1) {
    return inputSize - poolSize;
  } else {
    return (int64_t) ((index + sample) * alpha) - (int64_t) (sample * alpha);
  }
}

// We template on poolSizeW to allow the innermost loop to be unrolled
template <int64_t PoolSizeWStatic, typename Dtype, typename Acctype>
__global__ void SpatialFractionalMaxPooling_updateOutput(
  THCDeviceTensor<Dtype, 4> input,
  THCDeviceTensor<Dtype, 4> output,
  THCDeviceTensor<THCIndex_t, 4> indices,
  THCDeviceTensor<Dtype, 3> samples,
  int64_t poolSizeW, int64_t poolSizeH) {

  // Output (h, w) point that this thread is responsible for
  int64_t ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t plane = blockIdx.y;
  int64_t batch = blockIdx.z;

  // Each thread generates a specific output point
  if (ourOutputPoint < output.getSize(2) * output.getSize(3)) {
    int64_t outputW = ourOutputPoint % output.getSize(3);
    int64_t outputH = ourOutputPoint / output.getSize(3);

    int64_t poolW = getInterval<Dtype, Acctype>(ScalarConvert<Dtype, Acctype>::to(samples[batch][plane][0]), outputW,
                            input.getSize(3), output.getSize(3), poolSizeW);
    int64_t poolH = getInterval<Dtype, Acctype>(ScalarConvert<Dtype, Acctype>::to(samples[batch][plane][1]), outputH,
                            input.getSize(2), output.getSize(2), poolSizeH);

    Dtype maxVal = THCNumerics<Dtype>::min();
    int64_t maxIndex = -1;

    for (int64_t h = poolH; h < poolH + poolSizeH; ++h) {
      if (PoolSizeWStatic == -1) {
        for (int64_t w = poolW; w < poolW + poolSizeW; ++w) {
          Dtype val = input[batch][plane][h][w];
          // for consistency with THNN, favor the first max
          if (val > maxVal) {
            maxIndex = h * input.getSize(3) + w;
            maxVal = val;
          }
        }
      } else {
#pragma unroll
        for (int64_t i = 0; i < PoolSizeWStatic; ++i) {
          int64_t w = i + poolW;
          Dtype val = input[batch][plane][h][w];
          // for consistency with THNN, favor the first max
          if (val > maxVal) {
            maxIndex = h * input.getSize(3) + w;
            maxVal = val;
          }
        }
      }
    }

    assert(THCNumerics<Dtype>::ne(maxVal, THCNumerics<Dtype>::min()));
    assert(maxIndex != -1);

    // +1 for Lua index
    indices[batch][plane][outputH][outputW] = maxIndex + TH_INDEX_BASE;
    output[batch][plane][outputH][outputW] = maxVal;
  }
}

template <typename Dtype>
__global__ void SpatialFractionalMaxPooling_updateGradInput(
  THCDeviceTensor<Dtype, 4> gradInput,
  THCDeviceTensor<Dtype, 4> gradOutput,
  THCDeviceTensor<THCIndex_t, 4> indices) {
  // Output (h, w) point that this thread is responsible for
  int64_t ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t plane = blockIdx.y;
  int64_t batch = blockIdx.z;

  // Each thread generates a specific output point
  if (ourOutputPoint < gradOutput.getSize(2) * gradOutput.getSize(3)) {
    int64_t outputW = ourOutputPoint % gradOutput.getSize(3);
    int64_t outputH = ourOutputPoint / gradOutput.getSize(3);

    int64_t index = indices[batch][plane][outputH][outputW] - TH_INDEX_BASE;
    assert(index >= 0);
    int64_t inputW = index % gradInput.getSize(3);
    int64_t inputH = index / gradInput.getSize(3);
    assert(inputH < gradInput.getSize(2));

    atomicAdd(gradInput[batch][plane][inputH][inputW].data(),
              gradOutput[batch][plane][outputH][outputW]);
  }
}

#include "generic/SpatialFractionalMaxPooling.cu"
#include "THCGenerateFloatTypes.h"
