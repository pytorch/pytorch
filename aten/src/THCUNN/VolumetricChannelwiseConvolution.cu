// adopted from SpatialDepthwiseConvolution.cu
#include <THCUNN/THCUNN.h>
#include <THC/THCTensor.hpp>
#include <THC/THCDeviceTensor.cuh>
#include <THC/THCDeviceTensorUtils.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCReduceApplyUtils.cuh>
#include <THC/THCSortUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THCUNN/SharedMem.cuh>
#include <THCUNN/common.h>
#include <algorithm>


const int WARP_SIZE = 32;
// Crude benchmarks suggest 256 is better than 512 and 1024
// TODO: Autotune/use better heuristics, improve speed more.
const int MAX_BLOCK_SIZE = 256;

static int getGradParamsNumThreads(int batchSize){
//warp per item in a batch, up to a maximum
   return std::min(batchSize * WARP_SIZE, MAX_BLOCK_SIZE);
}

template <typename T, typename AccT, typename IndexType, int kSize>
__global__ void volumetricChannelwiseConvolutionUpdateOutput(
    const THCDeviceTensor<T, 5> input,
    THCDeviceTensor<T, 5> output,
    const THCDeviceTensor<T, 5> weight,
    const THCDeviceTensor<T, 1> bias,
    bool biasEnabled,
    IndexType totalElements,
    const int outputChannels,
    const int inputWidth, const int inputHeight, const int inputTime,
    const int outputWidth, const int outputHeight, const int outputTime,
    const int kernelWidth, const int kernelHeight, const int kernelTime,
    const int strideWidth, const int strideHeight, const int strideTime,
    const int padWidth, const int padHeight, const int padTime,
    const int dilationWidth, const int dilationHeight, const int dilationTime)
{
  const int KW_LIMIT = (kSize !=0) ? kSize : kernelWidth;
  const int KH_LIMIT = (kSize !=0) ? kSize : kernelHeight;
  const int KT_LIMIT = (kSize !=0) ? kSize : kernelTime;


  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // calculate n,c,l,h,w indices, replacing modulos by divide
    // and multiply add, result is same as would be in the code below
    // n = linearIndex / outputChannels / outputTime / outputHeight
    //      / outputWidth;
    // c = (linearIndex / outputTime / outputHeight / outputWidth)
    //      % outputChannels;
    // l = (linearIndex / outputWidth / outputHeight) % outputTime;
    // h = (linearIndex / outputWidth) % outputHeight;
    // w = linearIndex % outputWidth;

    int indtmp1 = linearIndex/outputWidth;
    const int w = linearIndex - indtmp1 * outputWidth;
    int indtmp2 = indtmp1/outputHeight;
    const int h = indtmp1 - indtmp2 * outputHeight;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1/outputTime;
    const int l = indtmp1 - indtmp2 * outputTime;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1/outputChannels;
    const int c = indtmp1 - indtmp2 * outputChannels;
    const int n = indtmp2;

    int inputChannel = c;
    int inputChannels = outputChannels;

    int weightOffset = c * kernelTime * kernelHeight * kernelWidth;

    AccT value = biasEnabled ? ScalarConvert<T, AccT>::to(bias.data()[c])
      : ScalarConvert<int, AccT>::to(0);
    const IndexType offset0 = (n * inputChannels + inputChannel) * inputTime
      * inputHeight * inputWidth;
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
    for (int kT = 0; kT < KT_LIMIT; ++kT) {
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
      for (int kH = 0; kH < KH_LIMIT; ++kH) {
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
        for (int kW = 0; kW < KW_LIMIT; ++kW) {
          const int l_in = -padTime + l * strideTime + kT * dilationTime;
          const int h_in = -padHeight + h * strideHeight + kH * dilationHeight;
          const int w_in = -padWidth + w * strideWidth + kW * dilationWidth;

          if ((l_in >= 0) && (l_in < inputTime) && (h_in >= 0) &&
            (h_in < inputHeight) && (w_in >= 0) && (w_in < inputWidth))
          {
            const IndexType offset =
              offset0 + (l_in * inputHeight + h_in) * inputWidth + w_in;
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
}

template <typename T, typename AccT, typename IndexType, int kSize, int stride>
__global__ void volumetricChannelwiseConvolutionUpdateGradInput(
    const THCDeviceTensor<T, 5> gradOutput,
    THCDeviceTensor<T, 5> gradInput,
    const THCDeviceTensor<T, 5> weight,
    IndexType totalElements,
    const int inputChannels,
    const int outputChannels,
    const int inputWidth, const int inputHeight, const int inputTime,
    const int outputWidth, const int outputHeight, const int outputTime,
    const int kernelWidth, const int kernelHeight, const int kernelTime,
    const int strideWidth, const int strideHeight, const int strideTime,
    const int padWidth, const int padHeight, const int padTime,
    const int dilationWidth, const int dilationHeight, const int dilationTime)
{

  const int KW_LIMIT = (kSize !=0) ? kSize : kernelWidth;
  const int KH_LIMIT = (kSize !=0) ? kSize : kernelHeight;
  const int KT_LIMIT = (kSize !=0) ? kSize : kernelTime;
  const int strideW = (stride !=0) ? stride : strideWidth;
  const int strideH = (stride !=0) ? stride : strideHeight;
  const int strideT = (stride !=0) ? stride : strideTime;

  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {

    int indtmp1 = linearIndex/inputWidth;
    const int w = linearIndex - indtmp1 * inputWidth;
    int indtmp2 = indtmp1/inputHeight;
    const int h = indtmp1 - indtmp2 * inputHeight;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1/inputTime;
    const int l = indtmp1 - indtmp2 * inputTime;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1/inputChannels;
    const int c = indtmp1 - indtmp2 * inputChannels;
    const int n = indtmp2;

    AccT value = ScalarConvert<int, AccT>::to(0);

    int weightOffset = c * kernelTime * kernelHeight * kernelWidth;
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
    for (int kT = 0; kT < KT_LIMIT; ++kT) {
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
      for (int kh = 0; kh < KH_LIMIT; ++kh) {
#ifdef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
        for (int kw = 0; kw < KW_LIMIT; ++kw) {
          int l_out = l + padTime - kT * dilationTime;
          int h_out = h + padHeight - kh * dilationHeight;
          int w_out = w + padWidth - kw * dilationWidth;
          if ((l_out % strideT == 0) && (h_out % strideH == 0)
              && (w_out % strideW == 0)) {
            l_out = l_out / strideT;
            h_out = h_out / strideH;
            w_out = w_out / strideW;

            if ((h_out >= 0) && (h_out < outputHeight)
                && (w_out >= 0) && (w_out < outputWidth)
                && (l_out >= 0) && (l_out < outputTime)) {

              const int offset =
                (((n * outputChannels + c) * outputTime + l_out)
                  * outputHeight + h_out) * outputWidth + w_out;
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
__global__ void volumetricChannelwiseConvolutionAccGradParameters(
    const THCDeviceTensor<T, 5> gradOutput,
    const THCDeviceTensor<T, 5> input,
    THCDeviceTensor<T, 5> gradWeight,
    const int batchSize,
    const int inputChannels,
    const int kernelChannels,
    const int inputWidth, const int inputHeight, const int inputTime,
    const int outputWidth, const int outputHeight, const int outputTime,
    const int kernelWidth, const int kernelHeight, const int kernelTime,
    const int strideWidth, const int strideHeight, const int strideTime,
    const int padWidth, const int padHeight, const int padTime,
    const int dilationWidth, const int dilationHeight, const int dilationTime)
{
  const int channelStride = kernelWidth * kernelHeight * kernelTime;

  // Have to use a statically typed Shared Memory pointer
  SharedMem<AccT> smem;

  // Each Block is responsible for accumulating over a permutation of
  // (channels x kH x kW), use blockIdx to determine which one
  int bidx = blockIdx.x;
  int kW = bidx % kernelWidth;
  int kH = (bidx / kernelWidth) % kernelHeight;
  int kT = (bidx / kernelWidth / kernelHeight) % kernelTime;
  int ch = (bidx / channelStride);

  // Need to calculate which input channel is associated with this filter
  // channel
  int inputCh = ch;

  AccT grad = ScalarConvert<float, AccT>::to(0.0);

  const int laneId = threadIdx.x % WARP_SIZE;
  const int batch = threadIdx.x / WARP_SIZE;
  const int nwarps = blockDim.x / WARP_SIZE;
  const int imageElements = outputWidth * outputHeight * outputTime;

  for (int batchIdx = batch; batchIdx < batchSize; batchIdx += nwarps){
    // Warp-stride loop over elements in a batch item
    for (IndexType idx = laneId; idx < imageElements; idx += WARP_SIZE) {
      int go_w_offset = idx % outputWidth;
      int go_h_offset = (idx / outputWidth) % outputHeight;
      int go_l_offset = (idx / outputWidth / outputHeight);

      int i_w_offset = (go_w_offset * strideWidth) + (kW * dilationWidth)
        - padWidth;
      int i_h_offset = (go_h_offset * strideHeight) + (kH * dilationHeight)
        - padHeight;
      int i_l_offset = (go_l_offset * strideTime) + (kT * dilationTime)
        - padTime;

      if (i_w_offset >= 0 && i_h_offset >= 0 && i_l_offset >= 0 &&
          i_w_offset < inputWidth && i_h_offset < inputHeight &&
          i_l_offset < inputTime) {
        int inputOffset = (((batchIdx * inputChannels + inputCh) * inputTime
          + i_l_offset) * inputHeight + i_h_offset) * inputWidth + i_w_offset;
        int outputOffset = (((batchIdx * kernelChannels + ch) * outputTime)
          * outputHeight ) * outputWidth + idx;
        grad = THCNumerics<AccT>::add(
            grad,
            THCNumerics<AccT>::mul(
              ScalarConvert<T, AccT>::to(input.data()[inputOffset]),
              ScalarConvert<T, AccT>::to(gradOutput.data()[outputOffset])));
      }
    }
  }
  __syncthreads();

  // At this point each thread in the block has a local gradient, which we need
  // to accumulate prior to writing the global value
  AccT *buf = smem.getPointer();
  AccT tval = reduceBlock<AccT, ReduceAdd<AccT>>(
    buf, blockDim.x, grad, ReduceAdd<AccT>(), ScalarConvert<float, AccT>::to(0)
  );

  // After reduction, first thread in the block has the gradient,
  // so its responsible for writing it to gradWeight
  if (threadIdx.x == 0) {
    int weightOffset = kW + kernelWidth * (kH + kernelHeight
      * (kT + kernelTime * ch));
    gradWeight.data()[weightOffset] = ScalarConvert<AccT, T>::to(tval);
  }
}

#include <THCUNN/generic/VolumetricChannelwiseConvolution.cu>
#include <THC/THCGenerateFloatTypes.h>
