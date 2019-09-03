// updateOutput, updateGradInput Kernels ported from Sergey Zagoruyko's pyinn, which itself was a
// port from Caffe

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
  const int in_height = inputHeight;
  const int in_width = inputWidth;
  const int in_depth = outputChannels;
  const int filter_height = kernelHeight;
  const int filter_width = kernelWidth;
  const int depth_multiplier = depthwiseMultiplier;
  const int stride = strideWidth;
  const int pad_height = padHeight;
  const int pad_width = padWidth;
  const int out_height = outputHeight;
  const int out_width = outputWidth;
  const int out_depth = outputChannels;

  for (IndexType thread_id = blockIdx.x * blockDim.x + threadIdx.x;
       thread_id < totalElements;
       thread_id += gridDim.x * blockDim.x) {
    // Compute the indexes of this thread in the output.
    //
    // We want coalesced reads so we make sure that each warp reads
    // a contiguous chunk of memory.
    //
    // THIS IS PROBABLY WRONG, we are not doing coalesced reads
    // into the input, because of the depth multiplier division...
    const int out_col = thread_id % out_width;
    const int out_row = (thread_id / out_width) % out_height;
    const int out_channel = (thread_id / out_width / out_height) % out_depth;
    const int batch = thread_id / out_width / out_height / out_depth;

    // Compute the input depth and the index of depth multiplier
    // based off the output depth index that this thread is
    // computing n.
    const int in_channel = out_channel / depth_multiplier;
    const int multiplier = out_channel % depth_multiplier;

    // Data is stored in the following format (let's assume we
    // flatten the height and width into one contiguous dimension
    // called "P".
    //
    // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
    //
    // Each row contains in_depth * in_height * in_width values
    // for each sample in the batch.
    //
    // We can further flatten it into:
    //
    // B1C1P1 B1C1P2 .....
    // B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 .....
    // B2C2P1 B2C2P2 ....
    //
    // where each row is a contiguous array of all of the spatial
    // pixels for a given batch and input depth.  The following
    // loop unrolls across the filter dimensions for a given thread,
    // indexing into the filter value and the corresponding input
    // patch.
    //
    // We can compute the index into the patch once right here.
    const int input_offset_temp =
        (batch * in_depth + in_channel) * (in_height * in_width);

    // Finally, we can iterate over the spatial dimensions and perform the
    // convolution, writing into the output at the end.
    //
    // We perform an additional optimization, where we can determine
    // whether the patch fits within the image indices statically, and
    // avoid boundary checking within the loop.
    const int input_row_start = out_row * stride - pad_height;
    const int input_col_start = out_col * stride - pad_width;
    const int input_row_end = input_row_start + filter_height;
    const int input_col_end = input_col_start + filter_width;

    AccT sum = ScalarConvert<int, AccT>::to(0);
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_row_end < in_height && input_col_end < in_width) {
      // Loop that doesn't need to check for boundary conditions.
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
      for (int filter_row = 0; filter_row < filter_height; ++filter_row) {
        const int in_row = input_row_start + filter_row;
        const int filter_offset_temp = filter_width * filter_row;
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
        for (int filter_col = 0; filter_col < filter_width; ++filter_col) {
          const int in_col = input_col_start + filter_col;

          const int input_offset =
              (input_offset_temp) + (in_row * in_width) + in_col;
          const int filter_offset =
              multiplier +
              depth_multiplier *
                  (in_channel + in_depth * (filter_col + filter_offset_temp));
          sum = THCNumerics<AccT>::add(
            sum,
            THCNumerics<AccT>::mul(
              ScalarConvert<T, AccT>::to(weight.data()[filter_offset]),
              ScalarConvert<T, AccT>::to(input.data()[input_offset])));
        }
      }
    } else {
      // Loop that needs to check for boundary conditions.
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
      for (int filter_row = 0; filter_row < filter_height; ++filter_row) {
        const int in_row = input_row_start + filter_row;
        const int filter_offset_temp = filter_width * filter_row;
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
        for (int filter_col = 0; filter_col < filter_width; ++filter_col) {
          const int in_col = input_col_start + filter_col;
          // TODO(vrv): the in_row check can be done outside of this loop;
          // benchmark both methods to determine the better decision.
          if (in_row >= 0 && in_row < in_height && in_col >= 0 &&
              in_col < in_width) {
            const int in_col = input_col_start + filter_col;

            // input_offset_temp indexes into the start of memory
            // where the spatial data starts.
            const int input_offset =
                (input_offset_temp) + (in_row * in_width) + in_col;

            const int filter_offset =
                multiplier +
                depth_multiplier *
                    (in_channel + in_depth * (filter_col + filter_offset_temp));
            sum = THCNumerics<AccT>::add(
              sum,
              THCNumerics<AccT>::mul(
                ScalarConvert<T, AccT>::to(weight.data()[filter_offset]),
                ScalarConvert<T, AccT>::to(input.data()[input_offset])));
          }
        }
      }
    }

    output.data()[thread_id] = static_cast<T>(sum);
  }
}

template <typename T, typename AccT, typename IndexType, int kSize, int stride>
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
  const int KW_LIMIT = (kSize !=0) ? kSize : kernelWidth;
  const int KH_LIMIT = (kSize !=0) ? kSize : kernelHeight;
  const int strideW = (stride !=0) ? stride : strideWidth;
  const int strideH = (stride !=0) ? stride : strideHeight;

  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {

    int indtmp1 = linearIndex/inputWidth;
    const int w = linearIndex - indtmp1 * inputWidth;
    int indtmp2 = indtmp1/inputHeight;
    const int h = indtmp1 - indtmp2 * inputHeight;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1/inputChannels;
    const int c = indtmp1 - indtmp2 * inputChannels;
    const int n = indtmp2;

    AccT value = ScalarConvert<int, AccT>::to(0);

#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
    for (int multiplier = 0; multiplier < depthwiseMultiplier; ++multiplier) {
      int och = (c * depthwiseMultiplier) + multiplier;
      int weightOffset = och * kernelHeight * kernelWidth;
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
      for (int kh = 0; kh < KH_LIMIT; ++kh) {
#ifdef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
        for (int kw = 0; kw < KW_LIMIT; ++kw) {
          int h_out = h + padHeight - kh * dilationHeight;
          int w_out = w + padWidth - kw * dilationWidth;
          if ((h_out % strideH == 0) && (w_out % strideW == 0)) {
            h_out = h_out / strideH;
            w_out = w_out / strideW;

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
  int ch = (bidx / channelStride);

  // Need to calculate which input channel is associated with this filter
  // channel
  int inputCh = ch / depthwiseMultiplier;

  AccT grad = ScalarConvert<float, AccT>::to(0.0);

  const int laneId = threadIdx.x % WARP_SIZE;
  const int batch = threadIdx.x / WARP_SIZE;
  const int nwarps = blockDim.x / WARP_SIZE;
  const int imageElements = outputWidth * outputHeight;
  // Use warp per item.  In the original kernel, a threadblock was used to sum over NHW.
  // Here, we use a warp to sum values over HW dimension, and if batchSize is larger than the
  // number of warps, a warp would loop over remaining batch items (e.g. if there are 8 warps,
  // warp 0 would go over 0-8-16 etc image, warp 1 over 1-9-17 etc). Later in blockReduce,
  // all the warps will be reduced anyway, thus the full reduction will be over NHW, like it
  // should be. That allows to get rid of one modulo operation inside the loop (because n/batchIdx
  // now does not have to be computed through modulo, you are just looping over it), and
  // bring a nice speed-up.
  for (int batchIdx = batch; batchIdx < batchSize; batchIdx += nwarps){
    // Warp-stride loop over elements in a batch item
    for (IndexType idx = laneId; idx < imageElements; idx += WARP_SIZE) {
    // Need to calculate the following: batch position, and offset into the gradOutput
    // in height, and width. We can intuit the corresponding position in the input from
    // the other parameters we have
      int go_w_offset = idx % outputWidth;
      int go_h_offset = (idx / outputWidth);

      int i_w_offset = (go_w_offset * strideWidth) + (kW * dilationWidth) - padWidth;
      int i_h_offset = (go_h_offset * strideHeight) + (kH * dilationHeight) - padHeight;

      if (i_w_offset >= 0 && i_h_offset >= 0 && i_w_offset < inputWidth && i_h_offset < inputHeight) {
        int inputOffset = ((batchIdx * inputChannels + inputCh) * inputHeight + i_h_offset) * inputWidth + i_w_offset;
        int outputOffset = ((batchIdx * kernelChannels + ch) * outputHeight ) * outputWidth + idx;
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
      buf, blockDim.x, grad, ReduceAdd<AccT>(), ScalarConvert<float, AccT>::to(0));

  // After reduction, first thread in the block has the gradient, so its responsible
  // for writing it to gradWeight
  if (threadIdx.x == 0) {
    int weightOffset = kW + (kernelWidth * kH) + (kernelWidth * kernelHeight * ch);
    gradWeight.data()[weightOffset] = ScalarConvert<AccT, T>::to(tval);
  }
}

#include <THCUNN/generic/SpatialDepthwiseConvolution.cu>
#include <THC/THCGenerateFloatTypes.h>
