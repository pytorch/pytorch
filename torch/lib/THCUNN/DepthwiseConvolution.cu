// Kernels ported from Sergey Zagoruyko's pyinn, which itself was a port from Caffe

#include "THCUNN.h"
#include "THCDeviceTensor.cuh"
#include "THCNumerics.cuh"

template <typename T, typename IndexType>
__global__ void depthwiseConvolutionUpdateOutput(
    const THCDeviceTensor<T, 4> input,
    THCDeviceTensor<T, 4> output,
    const THCDeviceTensor<T, 4> weight,
    IndexType totalElements,
    const int channels,
    const int inputWidth, const int inputHeight,
    const int outputWidth, const int outputHeight,
    const int kernelWidth, const int kernelHeight,
    const int strideWidth, const int strideHeight,
    const int padWidth, const int padHeight,
    const int dilationWidth, const int dilationHeight) {

  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    const int n = linearIndex / channels / outputHeight / outputWidth;
    const int c = (linearIndex / outputHeight / outputWidth) % channels;
    const int h = (linearIndex / outputWidth) % outputHeight;
    const int w = linearIndex % outputWidth;
    int weightOffset = c * kernelHeight * kernelWidth;
    T value = ScalarConvert<int, T>::to(0);

    for (int kH = 0; kH < kernelHeight; ++kH) {
      for (int kW = 0; kW < kernelWidth; ++kW) {
        const int h_in = -padHeight + h + strideHeight + kH * dilationHeight;
        const int w_in = -padWidth + w + strideWidth + kW * dilationWidth;

        if ((h_in >= 0) && (h_in < inputHeight) && (w_in >= 0) && (w_in < inputWidth)) {
          const IndexType offset = ((n * channels + c) * inputHeight + h_in) * inputWidth + w_in;
          value += weight.data()[weightOffset] * input.data()[offset];
        }

        ++weightOffset;
      }
    }

    output.data()[linearIndex] = value;
  }
}

template <typename T, typename IndexType>
__global__ void depthwiseConvolutionUpdateGradInput(
    const THCDeviceTensor<T, 4> gradOutput,
    THCDeviceTensor<T, 4> gradInput,
    const THCDeviceTensor<T, 4> weight,
    IndexType totalElements,
    const int channels,
    const int inputWidth, const int inputHeight,
    const int outputWidth, const int outputHeight,
    const int kernelWidth, const int kernelHeight,
    const int strideWidth, const int strideHeight,
    const int padWidth, const int padHeight,
    const int dilationWidth, const int dilationHeight) {

  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    const int n = linearIndex / channels / inputHeight / inputWidth;
    const int c = (linearIndex / inputHeight / inputWidth) % channels;
    const int h = (linearIndex / inputWidth) % inputHeight;
    const int w = linearIndex % inputWidth;
    int weightOffset = c * kernelHeight * kernelWidth;
    T value = 0;
    for (int kh = 0; kh < kernelHeight; ++kh) {
      for (int kw = 0; kw < kernelWidth; ++kw) {
        const int h_out_s = h + padHeight - kh * dilationHeight;
        const int w_out_s = w + padWidth - kw * dilationWidth;
        if (((h_out_s % strideHeight) == 0) && ((w_out_s % strideWidth) == 0)) {
          const int h_out = h_out_s / strideHeight;
          const int w_out = w_out_s / strideWidth;
          if ((h_out >= 0) && (h_out < outputHeight)
                && (w_out >= 0) && (w_out < outputWidth)) {
            const int offset = ((n * channels + c) * outputHeight + h_out)
                  * outputWidth + w_out;
            value += weight.data()[weightOffset] * gradOutput.data()[offset];
          }
        }
        ++weightOffset;
      }
    }
    gradInput.data()[linearIndex] = value;
  }
}

template <typename T, typename IndexType>
__global__ void depthwiseConvolutionAccGradParameters(
    const THCDeviceTensor<T, 4> gradOutput,
    const THCDeviceTensor<T, 4> input,
    THCDeviceTensor<T, 4> weightBuffer,
    IndexType totalElements,
    const int batchSize,
    const int channels,
    const int inputWidth, const int inputHeight,
    const int outputWidth, const int outputHeight,
    const int kernelWidth, const int kernelHeight,
    const int strideWidth, const int strideHeight,
    const int padWidth, const int padHeight,
    const int dilationWidth, const int dilationHeight) {

  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    const int h = (linearIndex / outputWidth) % outputHeight;
    const int w = linearIndex % outputWidth;
    const int kH = (linearIndex / kernelWidth / batchSize / outputHeight / outputWidth)
          % kernelHeight;
    const int kW = (linearIndex / batchSize / outputHeight / outputWidth) % kernelWidth;
    const int h_in = -padHeight + h * strideHeight + kH * dilationHeight;
    const int w_in = -padWidth + w * strideWidth + kW * dilationWidth;
    if ((h_in >= 0) && (h_in < inputHeight)
          && (w_in >= 0) && (w_in < inputWidth)) {
      const int c = linearIndex / kernelHeight / kernelWidth / batchSize / outputHeight / outputWidth;
      const int n = (linearIndex / outputHeight / outputWidth) % batchSize;
      const int gradOutputOffset = ((n * channels + c) * outputHeight + h)
            * outputWidth + w;
      const int inputOffset = ((n * channels + c) * inputHeight + h_in)
            * inputWidth + w_in;
      weightBuffer.data()[linearIndex] = gradOutput.data()[gradOutputOffset] * input.data()[inputOffset];
    } else {
      weightBuffer.data()[linearIndex] = 0;
    }
  }
}

#include "generic/DepthwiseConvolution.cu"
#include "THCGenerateFloatTypes.h"
