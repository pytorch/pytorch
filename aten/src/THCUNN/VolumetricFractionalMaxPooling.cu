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
template <int PoolSizeTStatic, typename Dtype, typename Acctype>
__global__ void VolumetricFractionalMaxPooling_updateOutput(
  THCDeviceTensor<Dtype, 5> input,
  THCDeviceTensor<Dtype, 5> output,
  THCDeviceTensor<THCIndex_t, 5> indices,
  THCDeviceTensor<Dtype, 3> samples,
  int poolSizeT, int poolSizeW, int poolSizeH) {

  // Output (h, w) point that this thread is responsible for
  int ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;

  // Each thread generates a specific output point
  if (ourOutputPoint < output.getSize(2) * output.getSize(3) * output.getSize(4)){
    int outputT = ourOutputPoint % output.getSize(4);
    int outputW = (ourOutputPoint / output.getSize(4)) % output.getSize(3);
    int outputH = ourOutputPoint / (output.getSize(3)*output.getSize(4));

    int poolT = getInterval<Dtype, Acctype>(ScalarConvert<Dtype, Acctype>::to(samples[batch][plane][0]), outputT,
                            input.getSize(4), output.getSize(4), poolSizeT);
    int poolW = getInterval<Dtype, Acctype>(ScalarConvert<Dtype, Acctype>::to(samples[batch][plane][1]), outputW,
                            input.getSize(3), output.getSize(3), poolSizeW);
    int poolH = getInterval<Dtype, Acctype>(ScalarConvert<Dtype, Acctype>::to(samples[batch][plane][2]), outputH,
                            input.getSize(2), output.getSize(2), poolSizeH);

    Dtype maxVal = THCNumerics<Dtype>::min();
    int maxIndex = -1;

    for (int h = poolH; h < poolH + poolSizeH; ++h) {
      for (int w = poolW; w < poolW + poolSizeW; ++w) {
        if (PoolSizeTStatic == -1) {
          for (int t = poolT; t < poolT + poolSizeT; ++t) {
            Dtype val = input[batch][plane][h][w][t];
            // for consistency with THNN, favor the first max
            if (val > maxVal) {
              maxIndex = h * input.getSize(3)*input.getSize(4) + w * input.getSize(4) + t;
              maxVal = val;
            }
          }
        } else {
#pragma unroll
          for (int i = 0; i < PoolSizeTStatic; ++i) {
            int t = i + poolT;
            Dtype val = input[batch][plane][h][w][t];
            // for consistency with THNN, favor the first max
            if (val > maxVal) {
              maxIndex = h * input.getSize(3)*input.getSize(4) + w * input.getSize(4) + t;
              maxVal = val;
            }
          }
        }
      }
    }

    assert(THCNumerics<Dtype>::ne(maxVal, THCNumerics<Dtype>::min()));
    assert(maxIndex != -1);

    // +1 for Lua index
    indices[batch][plane][outputH][outputW][outputT] = maxIndex + TH_INDEX_BASE;
    output[batch][plane][outputH][outputW][outputT] = maxVal;
  }
}

template <typename Dtype>
__global__ void VolumetricFractionalMaxPooling_updateGradInput(
  THCDeviceTensor<Dtype, 5> gradInput,
  THCDeviceTensor<Dtype, 5> gradOutput,
  THCDeviceTensor<THCIndex_t, 5> indices) {
  // Output (h, w) point that this thread is responsible for
  int ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;

  // Each thread generates a specific output point
  if (ourOutputPoint < gradOutput.getSize(2) * gradOutput.getSize(3) * gradOutput.getSize(4)) {
    int outputT = ourOutputPoint % gradOutput.getSize(4);
    int outputW = (ourOutputPoint / gradOutput.getSize(4)) % gradOutput.getSize(3);
    int outputH = ourOutputPoint / (gradOutput.getSize(3)*gradOutput.getSize(4));

    int index = indices[batch][plane][outputH][outputW][outputT] - TH_INDEX_BASE;
    assert(index >= 0);
    int inputT = index % gradInput.getSize(4);
    int inputW = (index / gradInput.getSize(4)) % gradInput.getSize(3);
    int inputH = index / (gradInput.getSize(3) * gradInput.getSize(4));
    assert(inputH < gradInput.getSize(2));

    atomicAdd(gradInput[batch][plane][inputH][inputW][inputT].data(),
              gradOutput[batch][plane][outputH][outputW][outputT]);
  }
}

#include "generic/VolumetricFractionalMaxPooling.cu"
#include "THCGenerateFloatTypes.h"
