// adopted from SpatialDepthwiseConvolution.cu
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCReduceApplyUtils.cuh>
#include <THCUNN/SharedMem.cuh>


#include <THC/THCDeviceTensor.cuh>
#include <THC/THCDeviceTensorUtils.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCSortUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THCUNN/common.h>
#include <algorithm>

namespace at {
namespace native {

namespace {

const int WARP_SIZE = 32;
// Crude benchmarks suggest 256 is better than 512 and 1024
// TODO: Autotune/use better heuristics, improve speed more.
const int MAX_BLOCK_SIZE = 256;

static int getGradParamsNumThreads(int batchSize){
//warp per item in a batch, up to a maximum
   return std::min(batchSize * WARP_SIZE, MAX_BLOCK_SIZE);
}

template <typename T, typename accT, typename IndexType, int kSize>
__global__ void volumetricChannelwiseConvolutionUpdateOutput(
    const Tensor& input,
    Tensor& output,
    const Tensor& weight,
    const Tensor& bias,
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

    int indtmp1 = linearIndex / outputWidth;
    const int w = linearIndex - indtmp1 * outputWidth;
    int indtmp2 = indtmp1 / outputHeight;
    const int h = indtmp1 - indtmp2 * outputHeight;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1/outputTime;
    const int l = indtmp1 - indtmp2 * outputTime;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1 / outputChannels;
    const int c = indtmp1 - indtmp2 * outputChannels;
    const int n = indtmp2;

    int inputChannel = c;
    int inputChannels = outputChannels;

    int weightOffset = c * kernelTime * kernelHeight * kernelWidth;

    accT value = biasEnabled ? ScalarConvert<T, accT>::to(bias.data()[c])
      : ScalarConvert<int, accT>::to(0);
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
            value = add(
              value,
              mul(
                ScalarConvert<T, accT>::to(weight.data()[weightOffset]),
                ScalarConvert<T, accT>::to(input.data()[offset])));
          }
          ++weightOffset;
        }
      }
      output.data()[linearIndex] = ScalarConvert<accT, T>::to(value);
    }
  }
}

template <typename T, typename accT, typename IndexType, int kSize, int stride>
__global__ void volumetricChannelwiseConvolutionUpdateGradInput(
    const Tensor& gradOutput,
    Tensor& gradInput,
    const Tensor& weight,
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

    accT value = ScalarConvert<int, accT>::to(0);

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
              value = add(
                value,
                mul(
                  ScalarConvert<T, accT>::to(weight.data()[weightOffset]),
                  ScalarConvert<T, accT>::to(gradOutput.data()[offset])));
            }
          }
          ++weightOffset;
        }
      }
    }
    gradInput.data()[linearIndex] = ScalarConvert<accT, T>::to(value);
  }
}


template <typename T, typename accT, typename IndexType>
__global__ void volumetricChannelwiseConvolutionAccGradParameters(
    const Tensor& gradOutput,
    const Tensor& input,
    Tensor& gradWeight,
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
  SharedMem<accT> smem;

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

  accT grad = ScalarConvert<float, accT>::to(0.0);

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
        grad = add(
            grad,
            mul(
              ScalarConvert<T, accT>::to(input.data()[inputOffset]),
              ScalarConvert<T, accT>::to(gradOutput.data()[outputOffset])));
      }
    }
  }
  __syncthreads();

  // At this point each thread in the block has a local gradient, which we need
  // to accumulate prior to writing the global value
  accT *buf = smem.getPointer();
  accT tval = reduceBlock<accT, ReduceAdd<accT>>(
    buf, blockDim.x, grad, ReduceAdd<accT>(), ScalarConvert<float, accT>::to(0)
  );

  // After reduction, first thread in the block has the gradient,
  // so its responsible for writing it to gradWeight
  if (threadIdx.x == 0) {
    int weightOffset = kW + kernelWidth * (kH + kernelHeight
      * (kT + kernelTime * ch));
    gradWeight.data()[weightOffset] = ScalarConvert<accT, T>::to(tval);
  }
}

// these two should be in the namespace
// I need to fill that one out 
Tensor& conv3d_channelwise3d_forward_out_cuda_template()
Tensor& conv3d_channelwise3d_backward_out_cuda_template()

} // namespace


// these are very much out
Tensor& conv3d_channelwise3d_forward_out_cuda(
  Tensor& output,
  const Tensor& input,
  IntList kernel_size,
  IntList stride_size,
  IntList pad_size,
  IntList dilation_size
)
{
  conv3d_channelwise3d_forward_out_cuda_template(
    output, input, kernel_size, stride_size, pad_size, dilation_size
  );
  return output;

}

Tensor conv3d_channelwise3d_forward_cuda(
  at::Tensor const& input,
  IntList kernel_size,
  IntList stride_size,
  IntList pad_size,
  IntList dilation_size
)
{
  auto output = at::empty({0}, input.options());
  conv3d_channelwise3d_forward_out_cuda_template(
    output, input, kernel_size, stride_size, pad_size, dilation_size
  );
  return output;
}

Tensor& conv3d_channelwise3d_backward_out_cuda(
  Tensor& gradInput,
  const Tensor& gradOutput,
  const Tensor& input,
  IntList kernel_size,
  IntList stride_size,
  IntList pad_size,
  IntList dilation_size
)
{
  conv3d_channelwise3d_backward_out_cuda_template(
    gradInput, gradOutput, kernel_size, stride_size, pad_size, dilation_size
  )
  return gradOutput;
}

Tensor& conv3d_channelwise3d_backward_cuda
const Tensor& gradOutput,
const Tensor& input,
IntList kernel_size,
IntList stride_size,
IntList pad_size,
IntList dilation_size
)
{
  auto gradInput = at::zeros_like(input);
  conv3d_channelwise3d_backward_out_cuda_template(
    gradInput, gradOutput, kernel_size, stride_size, pad_size, dilation_size
  )
  return gradOutput;
}

} // namespace at
} // namespace native
