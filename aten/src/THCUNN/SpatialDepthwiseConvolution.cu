// updateOutput, updateGradInput Kernels ported from Sergey Zagoruyko's pyinn, which itself was a
// port from Caffe

#include "THCUNN.h"
#include "THCTensor.hpp"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCNumerics.cuh"
#include "THCReduceApplyUtils.cuh"
#include "THCSortUtils.cuh"
#include "THCTensorMathReduce.cuh"
#include "SharedMem.cuh"
#include "common.h"
#include <algorithm>


const int64_t WARP_SIZE = 32;
// Crude benchmarks suggest 256 is better than 512 and 1024
// TODO: Autotune/use better heuristics, improve speed more.
const int64_t MAX_BLOCK_SIZE = 256;

static int64_t getGradParamsNumThreads(int64_t batchSize){
//warp per item in a batch, up to a maximum
   return std::min(batchSize * WARP_SIZE, MAX_BLOCK_SIZE);

}

template <typename T, typename AccT, typename IndexType, int64_t kSize>
__global__ void spatialDepthwiseConvolutionUpdateOutput(
    const THCDeviceTensor<T, 4> input,
    THCDeviceTensor<T, 4> output,
    const THCDeviceTensor<T, 4> weight,
    const THCDeviceTensor<T, 1> bias,
    bool biasEnabled,
    IndexType totalElements,
    const int64_t outputChannels,
    const int64_t depthwiseMultiplier,
    const int64_t inputWidth, const int64_t inputHeight,
    const int64_t outputWidth, const int64_t outputHeight,
    const int64_t kernelWidth, const int64_t kernelHeight,
    const int64_t strideWidth, const int64_t strideHeight,
    const int64_t padWidth, const int64_t padHeight,
    const int64_t dilationWidth, const int64_t dilationHeight)
{
  const int64_t KW_LIMIT = (kSize !=0) ? kSize : kernelWidth;
  const int64_t KH_LIMIT = (kSize !=0) ? kSize : kernelHeight;


  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    //calculate n,c,h,w indices, replacing modulos by divide and multiply add,
    //result is same as would be in the code below
    //const int64_t n = linearIndex / batchStride; //batchStride = outputChannels * outputHeight * outputWidth
    //const int64_t c = (linearIndex / channelStride) % outputChannels; //channelStride = outputHeight * outputWidth
    //const int64_t h = (linearIndex / outputWidth) % outputHeight;
    //const int64_t w = linearIndex % outputWidth;

    int64_t indtmp1 = linearIndex/outputWidth;
    const int64_t w = linearIndex - indtmp1 * outputWidth;
    int64_t indtmp2 = indtmp1/outputHeight;
    const int64_t h = indtmp1 - indtmp2 * outputHeight;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1/outputChannels;
    const int64_t c = indtmp1 - indtmp2 * outputChannels;
    const int64_t n = indtmp2;

    int64_t inputChannel = c;
    int64_t inputChannels = outputChannels;
    if (depthwiseMultiplier !=1) {
      inputChannel /= depthwiseMultiplier;
      inputChannels /= depthwiseMultiplier;
    }

    int64_t weightOffset = c * kernelHeight * kernelWidth;

    AccT value = biasEnabled ? ScalarConvert<T, AccT>::to(bias.data()[c]) : ScalarConvert<int, AccT>::to(0);
    const IndexType offset0 = (n * inputChannels + inputChannel) * inputHeight * inputWidth;
#pragma unroll
    for (int64_t kH = 0; kH < KH_LIMIT; ++kH) {
#pragma unroll
      for (int64_t kW = 0; kW < KW_LIMIT; ++kW) {
        const int64_t h_in = -padHeight + h * strideHeight + kH * dilationHeight;
        const int64_t w_in = -padWidth + w * strideWidth + kW * dilationWidth;

        if ((h_in >= 0) && (h_in < inputHeight) && (w_in >= 0) && (w_in < inputWidth)) {
          const IndexType offset = offset0 + h_in * inputWidth + w_in;
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

template <typename T, typename AccT, typename IndexType, int64_t kSize, int64_t stride>
__global__ void spatialDepthwiseConvolutionUpdateGradInput(
    const THCDeviceTensor<T, 4> gradOutput,
    THCDeviceTensor<T, 4> gradInput,
    const THCDeviceTensor<T, 4> weight,
    IndexType totalElements,
    const int64_t inputChannels,
    const int64_t depthwiseMultiplier,
    const int64_t outputChannels,
    const int64_t inputWidth, const int64_t inputHeight,
    const int64_t outputWidth, const int64_t outputHeight,
    const int64_t kernelWidth, const int64_t kernelHeight,
    const int64_t strideWidth, const int64_t strideHeight,
    const int64_t padWidth, const int64_t padHeight,
    const int64_t dilationWidth, const int64_t dilationHeight)
{
  const int64_t KW_LIMIT = (kSize !=0) ? kSize : kernelWidth;
  const int64_t KH_LIMIT = (kSize !=0) ? kSize : kernelHeight;
  const int64_t strideW = (stride !=0) ? stride : strideWidth;
  const int64_t strideH = (stride !=0) ? stride : strideHeight;

  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {

    int64_t indtmp1 = linearIndex/inputWidth;
    const int64_t w = linearIndex - indtmp1 * inputWidth;
    int64_t indtmp2 = indtmp1/inputHeight;
    const int64_t h = indtmp1 - indtmp2 * inputHeight;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1/inputChannels;
    const int64_t c = indtmp1 - indtmp2 * inputChannels;
    const int64_t n = indtmp2;

    AccT value = ScalarConvert<int, AccT>::to(0);

#pragma unroll
    for (int64_t multiplier = 0; multiplier < depthwiseMultiplier; ++multiplier) {
      int64_t och = (c * depthwiseMultiplier) + multiplier;
      int64_t weightOffset = och * kernelHeight * kernelWidth;
#pragma unroll
      for (int64_t kh = 0; kh < KH_LIMIT; ++kh) {
#pragma unroll
        for (int64_t kw = 0; kw < KW_LIMIT; ++kw) {
          int64_t h_out = h + padHeight - kh * dilationHeight;
          int64_t w_out = w + padWidth - kw * dilationWidth;
          if ((h_out % strideH == 0) && (w_out % strideW == 0)) {
            h_out = h_out / strideH;
            w_out = w_out / strideW;

            if ((h_out >= 0) && (h_out < outputHeight)
                  && (w_out >= 0) && (w_out < outputWidth)) {

              const int64_t offset = ((n * outputChannels + och) * outputHeight + h_out)
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
    const int64_t batchSize,
    const int64_t inputChannels,
    const int64_t kernelChannels,
    const int64_t depthwiseMultiplier,
    const int64_t inputWidth, const int64_t inputHeight,
    const int64_t outputWidth, const int64_t outputHeight,
    const int64_t kernelWidth, const int64_t kernelHeight,
    const int64_t strideWidth, const int64_t strideHeight,
    const int64_t padWidth, const int64_t padHeight,
    const int64_t dilationWidth, const int64_t dilationHeight)
{
  const int64_t channelStride = kernelWidth * kernelHeight;

  // Have to use a statically typed Shared Memory pointer
  SharedMem<AccT> smem;

  // Each Block is responsible for accumulating over a permutation of
  // (channels x kH x kW), use blockIdx to determine which one
  int64_t bidx = blockIdx.x;
  int64_t kW = bidx % kernelWidth;
  int64_t kH = (bidx / kernelWidth) % kernelHeight;
  int64_t ch = (bidx / channelStride);

  // Need to calculate which input channel is associated with this filter
  // channel
  int64_t inputCh = ch / depthwiseMultiplier;

  AccT grad = ScalarConvert<float, AccT>::to(0.0);

  const int64_t laneId = threadIdx.x % WARP_SIZE;
  const int64_t batch = threadIdx.x / WARP_SIZE;
  const int64_t nwarps = blockDim.x / WARP_SIZE;
  const int64_t imageElements = outputWidth * outputHeight;
  // Use warp per item.  In the original kernel, a threadblock was used to sum over NHW.
  // Here, we use a warp to sum values over HW dimension, and if batchSize is larger than the
  // number of warps, a warp would loop over remaining batch items (e.g. if there are 8 warps,
  // warp 0 would go over 0-8-16 etc image, warp 1 over 1-9-17 etc). Later in blockReduce,
  // all the warps will be reduced anyway, thus the full reduction will be over NHW, like it
  // should be. That allows to get rid of one modulo operation inside the loop (because n/batchIdx
  // now does not have to be computed through modulo, you are just looping over it), and
  // bring a nice speed-up.
  for (int64_t batchIdx = batch; batchIdx < batchSize; batchIdx += nwarps){
    // Warp-stride loop over elements in a batch item
    for (IndexType idx = laneId; idx < imageElements; idx += WARP_SIZE) {
    // Need to calculate the following: batch position, and offset into the gradOutput
    // in height, and width. We can intuit the corresponding position in the input from
    // the other parameters we have
      int64_t go_w_offset = idx % outputWidth;
      int64_t go_h_offset = (idx / outputWidth);

      int64_t i_w_offset = (go_w_offset * strideWidth) + (kW * dilationWidth) - padWidth;
      int64_t i_h_offset = (go_h_offset * strideHeight) + (kH * dilationHeight) - padHeight;

      if (i_w_offset >= 0 && i_h_offset >= 0 && i_w_offset < inputWidth && i_h_offset < inputHeight) {
        int64_t inputOffset = ((batchIdx * inputChannels + inputCh) * inputHeight + i_h_offset) * inputWidth + i_w_offset;
        int64_t outputOffset = ((batchIdx * kernelChannels + ch) * outputHeight ) * outputWidth + idx;
        grad = THCNumerics<AccT>::add(
            grad,
            THCNumerics<AccT>::mul(
              ScalarConvert<T, AccT>::to(input.data()[inputOffset]),
              ScalarConvert<T, AccT>::to(gradOutput.data()[outputOffset])));
      }
    }
  }
  __syncthreads();

  // At this point each thread in the block has a local gradient, which we need to
  // accumulate prior to writing the global value
  AccT *buf = smem.getPointer();
  AccT tval = reduceBlock<AccT, ReduceAdd<AccT>>(
      buf, blockDim.x, grad, ReduceAdd<AccT>(), ScalarConvert<int, AccT>::to(0));

  // After reduction, first thread in the block has the gradient, so its responsible
  // for writing it to gradWeight
  if (threadIdx.x == 0) {
    int64_t weightOffset = kW + (kernelWidth * kH) + (kernelWidth * kernelHeight * ch);
    gradWeight.data()[weightOffset] = ScalarConvert<AccT, T>::to(tval);
  }
}

#include "generic/SpatialDepthwiseConvolution.cu"
#include "THCGenerateFloatTypes.h"
