// adopted from SpatialDepthwiseConvolution.cu
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/native/DilatedConvolutionUtils.h>
#include <tuple>

#include <THC/THCReduceApplyUtils.cuh>
#include <THCUNN/SharedMem.cuh>


#include <THC/THCDeviceTensor.cuh>
#include <THC/THCDeviceTensorUtils.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCSortUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THCUNN/common.h>
#include <algorithm>

#include <stdio.h>
#include<iostream>


namespace at {
namespace native {

namespace {

const int WARP_SIZE = 32;
// Crude benchmarks suggest 256 is better than 512 and 1024
// TODO: Autotune/use better heuristics, improve speed more.
const int MAX_BLOCK_SIZE = 256;
const int MAX_THREADS = 1024;

static int getGradParamsNumThreads(int batchSize){
//warp per item in a batch, up to a maximum
   return std::min(batchSize * WARP_SIZE, MAX_BLOCK_SIZE);
}


// Your regular forward pass hopefully
// This code should not have changed since pull req 25437
template <typename T, typename accT, typename IndexType, int kSize>
__global__ void depthwiseConv3DOutput(
    const T * input, // 5D
    T * output, // 5D
    const T * weight,  // 5D
    const T * bias,  // 1D
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

    accT value = biasEnabled ? ScalarConvert<T, accT>::to(bias[c])
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
            value = THCNumerics<accT>::add(
              value,
              THCNumerics<accT>::mul(
                ScalarConvert<T, accT>::to(weight[weightOffset]),
                ScalarConvert<T, accT>::to(input[offset])));
          }
          ++weightOffset;
        } // for width
      } // for height
      output[linearIndex] = ScalarConvert<accT, T>::to(value);
    } // for time
  }
}

template <typename T, typename accT, typename IndexType, int kSize, int stride>
__global__ void depthwiseConv3dUpdateGradInput(
    const T * gradOutput,  // 5D
    T * gradInput,  // 5D
    const T * weight,  // 5D
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
              value = THCNumerics<accT>::add(
                value,
                THCNumerics<accT>::mul(
                  ScalarConvert<T, accT>::to(weight[weightOffset]),
                  ScalarConvert<T, accT>::to(gradOutput[offset])));
            }
          }
          ++weightOffset;
        }
      }
    }
    gradInput[linearIndex] = ScalarConvert<accT, T>::to(value);
  }
}


template <typename T, typename accT, typename IndexType>
__global__ void depthwiseConv3dUpdateGradWeight(
    const T * gradOutput,
    const T * input,
    T * gradWeight,
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
        grad = THCNumerics<accT>::add(
            grad,
            THCNumerics<accT>::mul(
              ScalarConvert<T, accT>::to(input[inputOffset]),
              ScalarConvert<T, accT>::to(gradOutput[outputOffset])));
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
    gradWeight[weightOffset] = ScalarConvert<accT, T>::to(tval);
  }
}


n  Tensor& output,
  const Tensor& input_,
  const Tensor& weight,
  const Tensor& bias,
  IntArrayRef kernel_size,
  IntArrayRef stride_size,
  IntArrayRef pad_size,
  IntArrayRef dilation_size) {
  
    TensorArg input_arg{input_, "input_", 1}, output_arg{output, "output", 2};

    checkAllSameGPU(
      "conv_depthwise3d_cuda_template", {input_arg, output_arg});

    
    for (int64_t i = 0; i < input_.ndimension(); i++) {
      TORCH_CHECK(
          input_.size(i) > 0,
          "conv_depthwise3d_cuda(): expected input to have non-empty spatial dimensions, "
          "but input has sizes ", input_.sizes(),
          " with dimension ", i, " being empty");
    }
    for (int64_t i = 0; i < weight.ndimension(); i++) {
      TORCH_CHECK(
          weight.size(i) > 0,
          "conv_depthwise3d_cuda(): expected weight to have non-empty dimensions, "
          "but weight has sizes ", weight.sizes(),
          " with dimension ", i, " being empty");
    }
    
    TORCH_CHECK(
        (input_.ndimension() == 5),
        "non-empty 5D tensor expected for input "
        "but input has size ", input_.ndimension());
      
    // Weight Tensor is shape (output_channels, 1, kH, kW)
    TORCH_CHECK(weight.size(1) == 1);

    // define kernel/stride/pad/dilation
    int64_t kW = kernel_size[2];
    int64_t kH = kernel_size[1];
    int64_t kT = kernel_size[0];

    int64_t dW = stride_size[2];
    int64_t dH = stride_size[1];
    int64_t dT = stride_size[0];

    int64_t padW = pad_size[2];
    int64_t padH = pad_size[1];
    int64_t padT = pad_size[0];

    int64_t dilationW = dilation_size[2];
    int64_t dilationH = dilation_size[1];
    int64_t dilationT = dilation_size[0];
    
    int64_t batch_size = input_.size(0);
    int64_t n_input_plane = input_.size(1);
    int64_t input_temp = input_.size(2);
    int64_t input_height = input_.size(3);
    int64_t input_width = input_.size(4);

    int batchSize = input_.size(0);
    int length = input_.size(2);
    int height = input_.size(3);
    int width = input_.size(4);

    int outputTime = (length + 2 * padT - (dilationT * (kT - 1) + 1)) / dT + 1;
    int outputHeight = (height + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
    int outputWidth = (width + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    int outputChannels = weight.size(0);
  
    // Pass empty contiguous tensor and resize it here
    output.resize_({batchSize, outputChannels, outputTime, outputHeight, outputWidth});

    // make sure input is contiguous
    const Tensor& input = input_.contiguous();

    int inputChannels = input.size(1);
    TORCH_CHECK(outputChannels == inputChannels);

    bool bias_flag = !bias.defined();

    // settring up the CUDA dispatch
    int n = output.numel();
    int blocks = GET_BLOCKS(n);
    dim3 grid(blocks);
    dim3 block(CUDA_NUM_THREADS);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input_.scalar_type(), "conv_depthwise3d_cuda", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();
        scalar_t* weight_data = weight.data_ptr<scalar_t>();
        scalar_t* bias_data = bias.data_ptr<scalar_t>();

        if (kW == 3 && kH == 3 && kT == 3) {
          depthwiseConv3DOutput<scalar_t, accscalar_t, unsigned int, 3><<<grid, block, 0, stream>>>(
            input_data, output_data,
            weight_data,
            bias_data,
            bias_flag, // if true, bias is not null
            n, // totalElements
            outputChannels, // ouptut_channels
            width, height, length, outputWidth, outputHeight, outputTime,
            kW, kH, kT, dW, dH, dT, padW, padH, padT, dilationW, dilationH, dilationT);
        } else if (kW == 1 && kH == 1 && kT == 1) {
          depthwiseConv3DOutput<scalar_t, accscalar_t, unsigned int, 1><<<grid, block, 0, stream>>>(
            input_data, output_data,
            weight_data,
            bias_data,
            bias_flag, // if true, bias is not null
            n, // totalElements
            outputChannels, // ouptut_channels
            width, height, length, outputWidth, outputHeight, outputTime,
            kW, kH, kT, dW, dH, dT, padW, padH, padT, dilationW, dilationH, dilationT);
        } else {
          depthwiseConv3DOutput<scalar_t, accscalar_t, unsigned int, 0><<<grid, block, 0, stream>>>(
            input_data, output_data,
            weight_data,
            bias_data,
            bias_flag, // if true, bias is not null
            n, // totalElements
            outputChannels, // ouptut_channels
            width, height, length, outputWidth, outputHeight, outputTime,
            kW, kH, kT, dW, dH, dT, padW, padH, padT, dilationW, dilationH, dilationT);
        }
    }); // A10 dispatch

    AT_CUDA_CHECK(cudaGetLastError());

} // conv_depthwise3d_cuda_template


void conv_depthwise3d_backward_input_cuda_template(
  Tensor& gradInput,
  const Tensor& gradOutput_,
  const Tensor& input,
  const Tensor& weight,
  IntArrayRef kernel_size,
  IntArrayRef stride_size,
  IntArrayRef pad_size,
  IntArrayRef dilation_size
) {

  TensorArg grad_input_arg{gradInput, "gradInput", 1};
  TensorArg grad_output_arg{gradOutput_, "gradOutput_", 2};
  TensorArg input_arg{input, "input", 3};

  checkAllSameGPU(
      "conv_depthwise3d_backward_input_cuda_template",
      {grad_input_arg, grad_output_arg, input_arg});

  // minimal shape checking
  TORCH_CHECK(
    input.size(0) == gradOutput_.size(0),
    "conv_depthwise3d_cuda(): expected batch sizes in backward to be the same"
    "but input has size(0) ", input.size(0),
    " and grad has ", gradOutput_.size(0));

  TORCH_CHECK(
    weight.size(0) == gradOutput_.size(1),
    "conv_depthwise3d_cuda(): expected number of channels in backward to be the same"
    "but weight has size(0) ", weight.size(0),
    " and grad has size(1)", gradOutput_.size(1));
  
  const Tensor gradOutput = gradOutput_.contiguous();

  gradInput.resize_as_(input);
  gradInput.zero_();

  int inputChannels = input.size(1);
  int length = input.size(2);
  int height = input.size(3);
  int width = input.size(4);

  int outputChannels = gradOutput.size(1);
  int outputTime = gradOutput.size(2);
  int outputHeight = gradOutput.size(3);
  int outputWidth = gradOutput.size(4);

  TORCH_CHECK(outputChannels == inputChannels,
    "conv_depthwise3d_backward_i_cuda(): expected inChannels and outChannels to be the same");
  

  // define kernel/stride/pad/dilation
  int64_t kernelSizeW = kernel_size[2];
  int64_t kernelSizeH = kernel_size[1];
  int64_t kernelSizeT = kernel_size[0];

  int64_t strideSizeW = stride_size[2];
  int64_t strideSizeH = stride_size[1];
  int64_t strideSizeT = stride_size[0];

  int64_t padSizeW = pad_size[2];
  int64_t padSizeH = pad_size[1];
  int64_t padSizeT = pad_size[0];

  int64_t dilSizeW = dilation_size[2];
  int64_t dilSizeH = dilation_size[1];
  int64_t dilSizeT = dilation_size[0];

  // upsample_3d_shape_check makes sure `nbatch != 0`
  int n = gradInput.numel();
  int blocks = GET_BLOCKS(n);
  dim3 grid(blocks);
  dim3 block(CUDA_NUM_THREADS);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.scalar_type(), "conv_depthwise3d_backward_input_cuda", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;

      scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
      scalar_t* gradOutput_data = gradOutput.data_ptr<scalar_t>();
      scalar_t* weight_data = weight.data_ptr<scalar_t>();
      if (kernelSizeW == 3 && kernelSizeH == 3 && kernelSizeT == 3) 
        if (strideSizeH == 1 && strideSizeW == 1 && strideSizeT == 1){
          depthwiseConv3dUpdateGradInput<scalar_t, accscalar_t, unsigned int, 3, 1><<<grid, block, 0, stream>>>(
              gradOutput_data, gradInput_data,
              weight_data,
              n, //total elements
              inputChannels,  // input channels
              outputChannels, // ouptut channels 
              width, height, length, outputWidth, outputHeight, outputTime,
              kernelSizeW, kernelSizeH, kernelSizeT,
              strideSizeW, strideSizeH, strideSizeT,
              padSizeW, padSizeH, padSizeT,
              dilSizeW, dilSizeH, dilSizeT);
        } else if (strideSizeH == 2 && strideSizeW == 2 && strideSizeT == 2){
          depthwiseConv3dUpdateGradInput<scalar_t, accscalar_t, unsigned int, 3, 2><<<grid, block, 0, stream>>>(
              gradOutput_data, gradInput_data,
              weight_data,
              n, //total elements
              inputChannels,  // input channels
              outputChannels, // ouptut channels 
              width, height, length, outputWidth, outputHeight, outputTime,
              kernelSizeW, kernelSizeH, kernelSizeT,
              strideSizeW, strideSizeH, strideSizeT,
              padSizeW, padSizeH, padSizeT,
              dilSizeW, dilSizeH, dilSizeT);
          } else {
            depthwiseConv3dUpdateGradInput<scalar_t, accscalar_t, unsigned int, 3, 0><<<grid, block, 0, stream>>>(
              gradOutput_data, gradInput_data,
              weight_data,
              n, //total elements
              inputChannels,  // input channels
              outputChannels, // ouptut channels 
              width, height, length, outputWidth, outputHeight, outputTime,
              kernelSizeW, kernelSizeH, kernelSizeT,
              strideSizeW, strideSizeH, strideSizeT,
              padSizeW, padSizeH, padSizeT,
              dilSizeW, dilSizeH, dilSizeT);          
          }
      else if (kernelSizeW == 1 && kernelSizeH == 1 && kernelSizeT == 1)
        if (strideSizeH == 1 && strideSizeW == 1 && strideSizeT == 1){
          depthwiseConv3dUpdateGradInput<scalar_t, accscalar_t, unsigned int, 1, 1><<<grid, block, 0, stream>>>(
            gradOutput_data, gradInput_data,
            weight_data,
            n, //total elements
            inputChannels,  // input channels
            outputChannels, // ouptut channels 
            width, height, length, outputWidth, outputHeight, outputTime,
            kernelSizeW, kernelSizeH, kernelSizeT,
            strideSizeW, strideSizeH, strideSizeT,
            padSizeW, padSizeH, padSizeT,
            dilSizeW, dilSizeH, dilSizeT);
        } else if (strideSizeH == 2 && strideSizeW == 2 && strideSizeT == 2){
          depthwiseConv3dUpdateGradInput<scalar_t, accscalar_t, unsigned int, 1, 2><<<grid, block, 0, stream>>>(
            gradOutput_data, gradInput_data,
            weight_data,
            n, //total elements
            inputChannels,  // input channels
            outputChannels, // ouptut channels 
            width, height, length, outputWidth, outputHeight, outputTime,
            kernelSizeW, kernelSizeH, kernelSizeT,
            strideSizeW, strideSizeH, strideSizeT,
            padSizeW, padSizeH, padSizeT,
            dilSizeW, dilSizeH, dilSizeT);
        } else {
          depthwiseConv3dUpdateGradInput<scalar_t, accscalar_t, unsigned int, 1, 0><<<grid, block, 0, stream>>>(
            gradOutput_data, gradInput_data,
            weight_data,
            n, //total elements
            inputChannels,  // input channels
            outputChannels, // ouptut channels 
            width, height, length, outputWidth, outputHeight, outputTime,
            kernelSizeW, kernelSizeH, kernelSizeT,
            strideSizeW, strideSizeH, strideSizeT,
            padSizeW, padSizeH, padSizeT,
            dilSizeW, dilSizeH, dilSizeT);       
        }
      else
      if (strideSizeH == 1 && strideSizeW == 1 && strideSizeT == 1){
        depthwiseConv3dUpdateGradInput<scalar_t, accscalar_t, unsigned int, 0, 1><<<grid, block, 0, stream>>>(
          gradOutput_data, gradInput_data,
          weight_data,
          n, //total elements
          inputChannels,  // input channels
          outputChannels, // ouptut channels 
          width, height, length, outputWidth, outputHeight, outputTime,
          kernelSizeW, kernelSizeH, kernelSizeT,
          strideSizeW, strideSizeH, strideSizeT,
          padSizeW, padSizeH, padSizeT,
          dilSizeW, dilSizeH, dilSizeT);
      } else if (strideSizeH == 2 && strideSizeW == 2 && strideSizeT == 2){
        depthwiseConv3dUpdateGradInput<scalar_t, accscalar_t, unsigned int, 0, 2><<<grid, block, 0, stream>>>(
          gradOutput_data, gradInput_data,
          weight_data,
          n, //total elements
          inputChannels,  // input channels
          outputChannels, // ouptut channels 
          width, height, length, outputWidth, outputHeight, outputTime,
          kernelSizeW, kernelSizeH, kernelSizeT,
          strideSizeW, strideSizeH, strideSizeT,
          padSizeW, padSizeH, padSizeT,
          dilSizeW, dilSizeH, dilSizeT);
      } else {
        depthwiseConv3dUpdateGradInput<scalar_t, accscalar_t, unsigned int, 0, 0><<<grid, block, 0, stream>>>(
          gradOutput_data, gradInput_data,
          weight_data,
          n, //total elements
          inputChannels,  // input channels
          outputChannels, // ouptut channels 
          width, height, length, outputWidth, outputHeight, outputTime,
          kernelSizeW, kernelSizeH, kernelSizeT,
          strideSizeW, strideSizeH, strideSizeT,
          padSizeW, padSizeH, padSizeT,
          dilSizeW, dilSizeH, dilSizeT);         
      }
    }); // AT DIspatch
  AT_CUDA_CHECK(cudaGetLastError());
} // conv_depthwise3d_backward_input_cuda_template


void conv_depthwise3d_backward_weight_cuda_template(
  Tensor& gradWeight,
  const Tensor& gradOutput_,
  const Tensor& input,
  const Tensor& weight,
  IntArrayRef kernel_size,
  IntArrayRef stride_size,
  IntArrayRef pad_size,
  IntArrayRef dilation_size
) {

  TensorArg grad_weight_arg{gradWeight, "gradWeight", 1};
  TensorArg grad_output_arg{gradOutput_, "gradOutput_", 2};
  TensorArg input_arg{input, "input", 3};

  checkAllSameGPU(
      "conv_depthwise3d_backward_weight_cuda_template",
      {grad_weight_arg, grad_output_arg, input_arg});

  // minimal shape checking
  TORCH_CHECK(
    input.size(0) == gradOutput_.size(0),
    "conv_depthwise3d_cuda(): expected batch sizes in backward weight to be the same"
    "but input has size(0) ", input.size(0),
    " and grad has ", gradOutput_.size(0));

  TORCH_CHECK(
    weight.size(0) == gradOutput_.size(1),
    "conv_depthwise3d_cuda(): expected number of channels in backward weight to be the same"
    "but weight has size(0) ", gradWeight.size(0),
    " and grad has size(1)", gradOutput_.size(1));
  
  // if checks are passed define tensors
  const Tensor gradOutput = gradOutput_.contiguous();

  int batchSize = input.size(0);
  int inputChannels = input.size(1);
  int length = input.size(2);
  int height = input.size(3);
  int width = input.size(4);

  int outputChannels = gradOutput.size(1);
  int outputTime = gradOutput.size(2);
  int outputHeight = gradOutput.size(3);
  int outputWidth = gradOutput.size(4);

  TORCH_CHECK(outputChannels == inputChannels,
    "conv_depthwise3d_backward_w_cuda(): expected inChannels and outChannels to be the same");


  gradWeight.resize_as_(weight);
  gradWeight.zero_();


  // define kernel/stride/pad/dilation
  int64_t kernelSizeW = kernel_size[2];
  int64_t kernelSizeH = kernel_size[1];
  int64_t kernelSizeT = kernel_size[0];

  int64_t strideSizeW = stride_size[2];
  int64_t strideSizeH = stride_size[1];
  int64_t strideSizeT = stride_size[0];

  int64_t padSizeW = pad_size[2];
  int64_t padSizeH = pad_size[1];
  int64_t padSizeT = pad_size[0];

  int64_t dilSizeW = dilation_size[2];
  int64_t dilSizeH = dilation_size[1];
  int64_t dilSizeT = dilation_size[0];

  // settring up the CUDA dispatch
  int blocks = outputChannels * kernelSizeH * kernelSizeT * kernelSizeW;
  
  dim3 grid(blocks);
  dim3 block(getGradParamsNumThreads(batchSize));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.scalar_type(), "conv_depthwise3d_backward_weight_cuda", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      scalar_t* gradWeight_data = gradWeight.data_ptr<scalar_t>();
      scalar_t* gradOutput_data = gradOutput.data_ptr<scalar_t>();
      scalar_t* input_data = input.data_ptr<scalar_t>();

      int smem = block.x * sizeof(accscalar_t);

      depthwiseConv3dUpdateGradWeight<scalar_t, accscalar_t, unsigned int><<<grid, block, smem, stream>>>(
          gradOutput_data, input_data, gradWeight_data,
          batchSize, inputChannels, outputChannels,
          width, height, length, outputWidth, outputHeight, outputTime,
          kernelSizeW, kernelSizeH, kernelSizeT,
          strideSizeW, strideSizeH, strideSizeT,
          padSizeW, padSizeH, padSizeT,
          dilSizeW, dilSizeH, dilSizeT);
    }); // AT Dispatch

  AT_CUDA_CHECK(cudaGetLastError());

} //conv_depthwise3d_backward_weight_cuda_template

} // namespace

Tensor _conv_depthwise3d_forward_cuda(
  Tensor& output,
  const Tensor& input,
  const Tensor& weight,
  const Tensor bias,
  IntArrayRef kernel_size,
  IntArrayRef stride_size,
  IntArrayRef pad_size,
  IntArrayRef dilation_size
)
{
  conv_depthwise3d_cuda_template(
    output, input, 
    weight, bias, 
    kernel_size, stride_size, pad_size, dilation_size);
  return output;

}

Tensor _conv3d_depthwise3d_backward_input_cuda(
  Tensor& gradInput,
  const Tensor& gradOutput,
  const Tensor& input,
  const Tensor& weight,
  IntArrayRef kernel_size,
  IntArrayRef stride_size,
  IntArrayRef pad_size,
  IntArrayRef dilation_size
)
{
  conv_depthwise3d_backward_input_cuda_template(
    gradInput, gradOutput, input, weight,
    kernel_size, stride_size, pad_size, dilation_size
  );
  return gradInput;
}


Tensor _conv3d_depthwise3d_backward_weight_cuda(
  Tensor& gradWeight,
  const Tensor& gradOutput,
  const Tensor& input,
  const Tensor& weight,
  IntArrayRef kernel_size,
  IntArrayRef stride_size,
  IntArrayRef pad_size,
  IntArrayRef dilation_size
)
{
  conv_depthwise3d_backward_weight_cuda_template(
    gradWeight, gradOutput, input, weight,
    kernel_size, stride_size, pad_size, dilation_size
  );
  return gradWeight;
}

// Publicly exposed functions

Tensor conv_depthwise3d_cuda(
  const Tensor& input, // b, c, n, h, w
  const Tensor& weight,
  const Tensor& bias,
  IntArrayRef kernel_size,
  IntArrayRef stride_size,
  IntArrayRef pad_size,
  IntArrayRef dilation_size) {

    auto options = input.options();
    Tensor output = at::empty({0}, options);
    _conv_depthwise3d_forward_cuda(
      output, input, weight, bias, kernel_size, stride_size, pad_size, dilation_size
    );
  return output;
}

std::tuple<at::Tensor,at::Tensor> conv_depthwise3d_backward_cuda(
  const Tensor& input,
  const Tensor& gradOutput,
  const Tensor& weight,
  IntArrayRef kernel_size,
  IntArrayRef stride_size,
  IntArrayRef pad_size,
  IntArrayRef dilation_size,
  std::array<bool,2> output_mask) {

    auto options = input.options();
    auto grad_input = at::empty({0}, options);
    options = weight.options();
    auto grad_weight = at::empty({0}, options);


    if (output_mask[0]) {
        _conv3d_depthwise3d_backward_input_cuda(
          grad_input, gradOutput, input, weight, 
          kernel_size, stride_size, pad_size, dilation_size
        );
    }
    if (output_mask[1]) {
      _conv3d_depthwise3d_backward_weight_cuda(
        grad_weight, gradOutput, input, weight,
        kernel_size, stride_size, pad_size, dilation_size
      );
    }
    return std::tuple<Tensor, Tensor>(grad_input, grad_weight);
}


} // namespace at
} // namespace native