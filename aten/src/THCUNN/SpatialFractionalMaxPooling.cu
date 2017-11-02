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
__device__ inline int getInterval(Acctype sample,
                                  int index,
                                  int inputSize,
                                  int outputSize,
                                  int poolSize) {
  Acctype alpha = (Acctype)(inputSize - poolSize) / (Acctype) (outputSize - 1);
  if (index == outputSize - 1) {
    return inputSize - poolSize;
  } else {
    return (int) ((index + sample) * alpha) - (int) (sample * alpha);
  }
}

// We template on poolSizeW to allow the innermost loop to be unrolled
template <int PoolSizeWStatic, typename Dtype, typename Acctype>
__global__ void SpatialFractionalMaxPooling_updateOutput(
  THCDeviceTensor<Dtype, 4> input,
  THCDeviceTensor<Dtype, 4> output,
  THCDeviceTensor<THCIndex_t, 4> indices,
  THCDeviceTensor<Dtype, 3> samples,
  int poolSizeW, int poolSizeH) {

  // Output (h, w) point that this thread is responsible for
  int ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;

  // Each thread generates a specific output point
  if (ourOutputPoint < output.getSize(2) * output.getSize(3)) {
    int outputW = ourOutputPoint % output.getSize(3);
    int outputH = ourOutputPoint / output.getSize(3);

    int poolW = getInterval<Dtype, Acctype>(ScalarConvert<Dtype, Acctype>::to(samples[batch][plane][0]), outputW,
                            input.getSize(3), output.getSize(3), poolSizeW);
    int poolH = getInterval<Dtype, Acctype>(ScalarConvert<Dtype, Acctype>::to(samples[batch][plane][1]), outputH,
                            input.getSize(2), output.getSize(2), poolSizeH);

    Dtype maxVal = THCNumerics<Dtype>::min();
    int maxIndex = -1;

    for (int h = poolH; h < poolH + poolSizeH; ++h) {
      if (PoolSizeWStatic == -1) {
        for (int w = poolW; w < poolW + poolSizeW; ++w) {
          Dtype val = input[batch][plane][h][w];
          // for consistency with THNN, favor the first max
          if (val > maxVal) {
            maxIndex = h * input.getSize(3) + w;
            maxVal = val;
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < PoolSizeWStatic; ++i) {
          int w = i + poolW;
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
  int ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;

  // Each thread generates a specific output point
  if (ourOutputPoint < gradOutput.getSize(2) * gradOutput.getSize(3)) {
    int outputW = ourOutputPoint % gradOutput.getSize(3);
    int outputH = ourOutputPoint / gradOutput.getSize(3);

    int index = indices[batch][plane][outputH][outputW] - TH_INDEX_BASE;
    assert(index >= 0);
    int inputW = index % gradInput.getSize(3);
    int inputH = index / gradInput.getSize(3);
    assert(inputH < gradInput.getSize(2));

    atomicAdd(gradInput[batch][plane][inputH][inputW].data(),
              gradOutput[batch][plane][outputH][outputW]);
  }
}

#include "generic/SpatialFractionalMaxPooling.cu"
#include "THCGenerateFloatTypes.h"
