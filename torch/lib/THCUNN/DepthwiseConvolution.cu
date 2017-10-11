// Kernels ported from Sergey Zagoruyko's pyinn, which itself was a port from Caffe

#include "THCUNN.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCNumerics.cuh"
#include "THCReduceApplyUtils.cuh"
#include "THCTensorMathReduce.cuh"
#include "SharedMem.cuh"
#include "common.h"

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

  /* if (threadIdx.x == 0) { */
  /*   printf("Params - output_nelem: %d, channels: %d, w: %d, h: %d, oW: %d, oH: %d, kW: %d, kH: %d\n", */
  /*       totalElements, channels, inputWidth, inputHeight, outputWidth, outputHeight, kernelWidth, kernelHeight); */
  /* } */

  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    const int n = linearIndex / channels / outputHeight / outputWidth;
    const int c = (linearIndex / outputHeight / outputWidth) % channels;
    const int h = (linearIndex / outputWidth) % outputHeight;
    const int w = linearIndex % outputWidth;

    /* printf("calculating for (n = %d, c = %d, h = %d, w = %d\n", n, c, h, w); */

    int weightOffset = c * kernelHeight * kernelWidth;

    T value = ScalarConvert<int, T>::to(0);

    for (int kH = 0; kH < kernelHeight; ++kH) {
      for (int kW = 0; kW < kernelWidth; ++kW) {
        const int h_in = -padHeight + h * strideHeight + kH * dilationHeight;
        const int w_in = -padWidth + w * strideWidth + kW * dilationWidth;

        /* printf("h_in %d, w_in %d\n", h_in, w_in); */

        if ((h_in >= 0) && (h_in < inputHeight) && (w_in >= 0) && (w_in < inputWidth)) {
          const IndexType offset = ((n * channels + c) * inputHeight + h_in) * inputWidth + w_in;
          /* printf("multiplying input offset: %d with weight offset: %d\n", offset, weightOffset); */
          value = THCNumerics<T>::add(
            value,
            THCNumerics<T>::mul(weight.data()[weightOffset], input.data()[offset]));
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

  /* if (threadIdx.x == 0) { */
  /*   printf("Params - gradInput_nelem: %d, channels: %d, w: %d, h: %d, oW: %d, oH: %d, kW: %d, kH: %d\n", */
  /*       totalElements, channels, inputWidth, inputHeight, outputWidth, outputHeight, kernelWidth, kernelHeight); */
  /* } */

  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    const int n = linearIndex / channels / inputHeight / inputWidth;
    const int c = (linearIndex / inputHeight / inputWidth) % channels;
    const int h = (linearIndex / inputWidth) % inputHeight;
    const int w = linearIndex % inputWidth;

    /* printf("calculating for (n = %d, c = %d, h = %d, w = %d\n)", n, c, h, w); */

    int weightOffset = c * kernelHeight * kernelWidth;
    T value = ScalarConvert<int, T>::to(0);

    for (int kh = 0; kh < kernelHeight; ++kh) {
      for (int kw = 0; kw < kernelWidth; ++kw) {
        const int h_out_s = h + padHeight - kh * dilationHeight;
        const int w_out_s = w + padWidth - kw * dilationWidth;

        if (((h_out_s % strideHeight) == 0) && ((w_out_s % strideWidth) == 0)) {
          const int h_out = h_out_s / strideHeight;
          const int w_out = w_out_s / strideWidth;

          if ((h_out >= 0) && (h_out < outputHeight)
                && (w_out >= 0) && (w_out < outputWidth)) {
            /* printf("gradOutput h %d, w %d\n", h_out, w_out); */

            const int offset = ((n * channels + c) * outputHeight + h_out)
                  * outputWidth + w_out;
            value = THCNumerics<T>::add(
              value,
              THCNumerics<T>::mul(weight.data()[weightOffset], gradOutput.data()[offset]));
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
    THCDeviceTensor<T, 4> gradWeight,
    const int batchSize,
    const int channels,
    IndexType blockElements,
    const int inputWidth, const int inputHeight,
    const int outputWidth, const int outputHeight,
    const int kernelWidth, const int kernelHeight,
    const int strideWidth, const int strideHeight,
    const int padWidth, const int padHeight,
    const int dilationWidth, const int dilationHeight)
{
  /* if (blockIdx.x == 0 && threadIdx.x == 0) { */
  /*   printf("Params - block nelem: %d, channels: %d, w: %d, h: %d, oW: %d, oH: %d, kW: %d, kH: %d\n", */
  /*       blockElements, channels, inputWidth, inputHeight, outputWidth, outputHeight, kernelWidth, kernelHeight); */
  /* } */

  /* for (int bidx = 0; bidx < blockElements; ++bidx) { */

  SharedMem<T> smem;
  int bidx = blockIdx.x;

    // Each Block is responsible for accumulating over a permutation of
    // (channels x kH x kW)
    int kW = bidx % kernelWidth;
    int kH = (bidx / kernelWidth) % kernelHeight;
    int ch = (bidx / kernelWidth / kernelHeight) % channels;

    /* printf("calculating grad weight for C %d, kH %d, kW %d\n", ch, kH, kW); */

    T grad = ScalarConvert<float, T>::to(0.0);

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
        grad = THCNumerics<T>::add(
            grad,
            THCNumerics<T>::mul(
              input[batch][ch][i_h_offset].data()[i_w_offset],
              gradOutput[batch][ch][go_h_offset].data()[go_w_offset]));
      }

      /* printf("idx %d, go_h %d, go_w %d, batch %d\n", idx, go_h_offset, go_w_offset, batch); */

      /* printf("multiplying input[batch %d][ch %d][h %d][w %d] with gradOutput[ch %d][oH %d][oW %d]\n", */
      /*     batch, ch, go_h_offset + kH, go_w_offset + kW, ch, go_h_offset, go_w_offset); */

    }
    __syncthreads();

    // At this point each thread in the block has a local gradient, which we need to
    // accumulate prior to writing the global value
    T *buf = smem.getPointer();
    T tval = reduceBlock<T, ReduceAdd<T, T>>(buf, blockDim.x, grad, ReduceAdd<T, T>(), ScalarConvert<float, T>::to(0));

    // After reduction, first thread in the block has the gradient, so its responsible
    // for writing it to gradWeight
    if (threadIdx.x == 0) {
      /* printf("tval %f\n", tval); */
      int offset = kW + (kernelWidth * kH) + (kernelWidth * kernelHeight * ch);
      /* printf("offset: %d\n", kW + (kernelWidth * kH) + (kernelWidth * kernelHeight * ch)); */
      gradWeight.data()[offset] = tval;
    }
  /* } */
}

#include "generic/DepthwiseConvolution.cu"
#include "THCGenerateFloatTypes.h"
