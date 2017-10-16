// updateOutput, updateGradInput Kernels ported from Sergey Zagoruyko's pyinn, which itself was a
// port from Caffe

#include "THCUNN.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCNumerics.cuh"
#include "THCReduceApplyUtils.cuh"
#include "THCSortUtils.cuh"
#include "THCTensorMathReduce.cuh"
#include "SharedMem.cuh"
#include "common.h"

template <typename T, typename AccT, typename IndexType>
__global__ void spatialDepthwiseConvolutionUpdateOutput(
    const THCDeviceTensor<T, 4> input,
    THCDeviceTensor<T, 4> output,
    const THCDeviceTensor<T, 4> weight,
    const THCDeviceTensor<T, 1> bias,
    bool biasEnabled,
    IndexType totalElements,
    const int outputChannels,
    const int depthwiseMultiplier,
    const int inputWidth, const int inputHeight,
    const int outputWidth, const int outputHeight,
    const int kernelWidth, const int kernelHeight,
    const int strideWidth, const int strideHeight,
    const int padWidth, const int padHeight,
    const int dilationWidth, const int dilationHeight)
{
  const int channelStride = outputHeight * outputWidth;
  const int batchStride = outputChannels * channelStride;

  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {

    const int n = linearIndex / batchStride;
    const int c = (linearIndex / channelStride) % outputChannels;
    const int h = (linearIndex / outputWidth) % outputHeight;
    const int w = linearIndex % outputWidth;

    const int inputChannel = c / depthwiseMultiplier;
    const int inputChannels = outputChannels / depthwiseMultiplier;

    int weightOffset = c * kernelHeight * kernelWidth;

    AccT value = biasEnabled ? ScalarConvert<T, AccT>::to(bias.data()[c]) : ScalarConvert<int, AccT>::to(0);
    for (int kH = 0; kH < kernelHeight; ++kH) {
      for (int kW = 0; kW < kernelWidth; ++kW) {
        const int h_in = -padHeight + h * strideHeight + kH * dilationHeight;
        const int w_in = -padWidth + w * strideWidth + kW * dilationWidth;

        if ((h_in >= 0) && (h_in < inputHeight) && (w_in >= 0) && (w_in < inputWidth)) {
          const IndexType offset = ((n * inputChannels + inputChannel) * inputHeight + h_in) *
                                    inputWidth + w_in;
          value = THCNumerics<AccT>::add(
            value,
            THCNumerics<AccT>::mul(
              ScalarConvert<T, AccT>::to(weight.data()[weightOffset]),
              ScalarConvert<T, AccT>::to(input.data()[offset])));
        }
        ++weightOffset;
      }
    }
    output.data()[linearIndex] = ScalarConvert<AccT, T>::to(value);
  }
}

template <typename T, typename AccT, typename IndexType>
__global__ void spatialDepthwiseConvolutionUpdateGradInput(
    const THCDeviceTensor<T, 4> gradOutput,
    THCDeviceTensor<T, 4> gradInput,
    const THCDeviceTensor<T, 4> weight,
    IndexType totalElements,
    const int inputChannels,
    const int depthwiseMultiplier,
    const int outputChannels,
    const int inputWidth, const int inputHeight,
    const int outputWidth, const int outputHeight,
    const int kernelWidth, const int kernelHeight,
    const int strideWidth, const int strideHeight,
    const int padWidth, const int padHeight,
    const int dilationWidth, const int dilationHeight)
{
  const int channelStride = inputHeight * inputWidth;
  const int batchStride = inputChannels * channelStride;

  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {

    const int n = linearIndex / batchStride;
    const int c = (linearIndex / channelStride) % inputChannels;
    const int h = (linearIndex / inputWidth) % inputHeight;
    const int w = linearIndex % inputWidth;

    AccT value = ScalarConvert<int, AccT>::to(0);
    for (int multiplier = 0; multiplier < depthwiseMultiplier; ++multiplier) {
      int och = (c * depthwiseMultiplier) + multiplier;
      int weightOffset = och * kernelHeight * kernelWidth;
      for (int kh = 0; kh < kernelHeight; ++kh) {
        for (int kw = 0; kw < kernelWidth; ++kw) {
          const int h_out_s = h + padHeight - kh * dilationHeight;
          const int w_out_s = w + padWidth - kw * dilationWidth;

          if (((h_out_s % strideHeight) == 0) && ((w_out_s % strideWidth) == 0)) {
            const int h_out = h_out_s / strideHeight;
            const int w_out = w_out_s / strideWidth;

            if ((h_out >= 0) && (h_out < outputHeight)
                  && (w_out >= 0) && (w_out < outputWidth)) {

              const int offset = ((n * outputChannels + och) * outputHeight + h_out)
                    * outputWidth + w_out;
              value = THCNumerics<AccT>::add(
                value,
                THCNumerics<AccT>::mul(
                  ScalarConvert<T, AccT>::to(weight.data()[weightOffset]),
                  ScalarConvert<T, AccT>::to(gradOutput.data()[offset])));
            }
          }
          ++weightOffset;
        }
      }
    }
    gradInput.data()[linearIndex] = ScalarConvert<AccT, T>::to(value);
  }
}

template <typename T, typename AccT, typename IndexType>
__global__ void spatialDepthwiseConvolutionAccGradParameters(
    const THCDeviceTensor<T, 4> gradOutput,
    const THCDeviceTensor<T, 4> input,
    THCDeviceTensor<T, 4> gradWeight,
    const int batchSize,
    const int inputChannels,
    const int kernelChannels,
    const int depthwiseMultiplier,
    IndexType blockElements,
    const int inputWidth, const int inputHeight,
    const int outputWidth, const int outputHeight,
    const int kernelWidth, const int kernelHeight,
    const int strideWidth, const int strideHeight,
    const int padWidth, const int padHeight,
    const int dilationWidth, const int dilationHeight)
{
  const int channelStride = kernelWidth * kernelHeight;

  // Have to use a statically typed Shared Memory pointer
  SharedMem<AccT> smem;

  // Each Block is responsible for accumulating over a permutation of
  // (channels x kH x kW), use blockIdx to determine which one
  int bidx = blockIdx.x;
  int kW = bidx % kernelWidth;
  int kH = (bidx / kernelWidth) % kernelHeight;
  int ch = (bidx / channelStride) % kernelChannels;

  // Need to calculate which input channel is associated with this filter
  // channel
  int inputCh = ch / depthwiseMultiplier;

  AccT grad = ScalarConvert<float, AccT>::to(0.0);

  // Block-stride loop over the number of elements we need to reduce
  for (IndexType idx = threadIdx.x; idx < blockElements; idx += blockDim.x) {
    // Need to calculate the following: batch position, and offset into the gradOutput
    // in height, and width. We can intuit the corresponding position in the input from
    // the other parameters we have
    int go_w_offset = idx % outputWidth;
    int go_h_offset = (idx / outputWidth) % outputHeight;
    int batch = (idx / outputWidth / outputHeight) % batchSize;

    int i_w_offset = (go_w_offset * strideWidth) + (kW * dilationWidth) - padWidth;
    int i_h_offset = (go_h_offset * strideHeight) + (kH * dilationHeight) - padHeight;

    if (i_w_offset >= 0 && i_h_offset >= 0 && i_w_offset < inputWidth && i_h_offset < inputHeight) {
      int inputOffset = ((batch * inputChannels + inputCh) * inputHeight + i_h_offset) * inputWidth + i_w_offset;
      int outputOffset = ((batch * kernelChannels + ch) * outputHeight + go_h_offset) * outputWidth + go_w_offset;
      grad = THCNumerics<AccT>::add(
          grad,
          THCNumerics<AccT>::mul(
            ScalarConvert<T, AccT>::to(input.data()[inputOffset]),
            ScalarConvert<T, AccT>::to(gradOutput.data()[outputOffset])));
    }
  }
  __syncthreads();

  // At this point each thread in the block has a local gradient, which we need to
  // accumulate prior to writing the global value
  AccT *buf = smem.getPointer();
  AccT tval = reduceBlock<AccT, ReduceAdd<AccT, AccT>>(
      buf, blockDim.x, grad, ReduceAdd<AccT, AccT>(), ScalarConvert<float, AccT>::to(0));

  // After reduction, first thread in the block has the gradient, so its responsible
  // for writing it to gradWeight
  if (threadIdx.x == 0) {
    int weightOffset = kW + (kernelWidth * kH) + (kernelWidth * kernelHeight * ch);
    gradWeight.data()[weightOffset] = ScalarConvert<AccT, T>::to(tval);
  }
}

#include "generic/SpatialDepthwiseConvolution.cu"
#include "THCGenerateFloatTypes.h"
