#include <THCUNN/THCUNN.h>
#include <THCUNN/common.h>
#include <THC/THCTensor.hpp>

#include <THCUNN/upsampling.h>
#include <THC/THCDeviceTensor.cuh>
#include <THC/THCDeviceTensorUtils.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCAtomics.cuh>

template<typename Dtype, typename Acctype>
__global__ void nearest_neighbor_5d_kernel(
		const int n,
		const THCDeviceTensor<Dtype, 5> data1,
		THCDeviceTensor<Dtype, 5> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int depth1 = data1.getSize(2);
  const int height1 = data1.getSize(3);
  const int width1 = data1.getSize(4);
  const int depth2 = data2.getSize(2);
  const int height2 = data2.getSize(3);
  const int width2 = data2.getSize(4);
  const float depth_scale = (float) depth1 / (float) depth2;
  const float height_scale = (float) height1 / (float) height2;
  const float width_scale = (float) width1 / (float) width2;

  if (index < n) {
    const int w2 = (index % (height2*width2)) % width2; // 0:width2-1
    const int h2 = (index % (height2*width2)) / width2; // 0:height2-1
    const int d2 = index / (height2*width2);            // 0:depth2-1
    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int d1 = d2;
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data1[n][c][d1][h1][w1];
          data2[n][c][d2][h2][w2] = val;
        }
      }
      return;
    }
    //
    const int h1 = nearest_neighbor_compute_source_index(height_scale, h2, height1);
    const int w1 = nearest_neighbor_compute_source_index(width_scale, w2, width1);
    const int d1 = nearest_neighbor_compute_source_index(depth_scale, d2, depth1);
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
	const Dtype val = data1[n][c][d1][h1][w1];
	data2[n][c][d2][h2][w2] = val;
      }
    }
  }
}

// Backward operation
template <typename Dtype, typename Acctype>
__global__ void nearest_neighbor_5d_kernel_backward(
		const int n,
		THCDeviceTensor<Dtype, 5> data1,
		const THCDeviceTensor<Dtype, 5> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int depth1 = data1.getSize(2);
  const int height1 = data1.getSize(3);
  const int width1 = data1.getSize(4);
  const int depth2 = data2.getSize(2);
  const int height2 = data2.getSize(3);
  const int width2 = data2.getSize(4);
  const float depth_scale = (float) depth1 / (float) depth2;
  const float height_scale = (float) height1 / (float) height2;
  const float width_scale = (float) width1 / (float) width2;

  if (index < n) {
    const int w2 = (index % (height2*width2)) % width2; // 0:width2-1
    const int h2 = (index % (height2*width2)) / width2; // 0:height2-1
    const int d2 = index / (height2*width2);            // 0:depth2-1

    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int d1 = d2;
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data2[n][c][d1][h1][w1];
          data1[n][c][d2][h2][w2] = val;
        }
      }
      return;
    }
    //
    const int h1 = nearest_neighbor_compute_source_index(height_scale, h2, height1);
    const int w1 = nearest_neighbor_compute_source_index(width_scale, w2, width1);
    const int d1 = nearest_neighbor_compute_source_index(depth_scale, d2, depth1);
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
	const Dtype val = data2[n][c][d2][h2][w2];
	atomicAdd(data1[n][c][d1][h1][w1].data(), val);
      }
    }
  }
}


#include <THCUNN/generic/VolumetricUpSamplingNearest.cu>
#include <THC/THCGenerateFloatTypes.h>

//---------

#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/VolumetricUpSamplingNearest.cu"
#else

#include <THCUNN/common.h>
#include "ATen/cuda/CUDAContext.h"

static inline void THNN_(VolumetricUpSamplingNearest_shapeCheck)
                        (THCState *state,
                         THCTensor *input, THCTensor *gradOutput,
                         int nBatch, int nChannels,
                         int inputDepth, int inputHeight, int inputWidth,
                         int outputDepth, int outputHeight, int outputWidth) {
  THArgCheck(inputDepth > 0 && inputHeight > 0 && inputWidth > 0
             && outputDepth && outputHeight > 0 && outputWidth > 0, 2,
             "input and output sizes should be greater than 0,"
             " but got input (D: %d, H: %d, W: %d) output (D: %d, H: %d, W: %d)",
             inputDepth, inputHeight, inputWidth, outputDepth, outputHeight, outputWidth);
  if (input != NULL) {
     THCUNN_argCheck(state, THTensor_nDimensionLegacyAll(input) == 5, 2, input,
                     "5D input tensor expected but got: %s");
  }

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, 5, 0, nBatch);
    THCUNN_check_dim_size(state, gradOutput, 5, 1, nChannels);
    THCUNN_check_dim_size(state, gradOutput, 5, 2, outputDepth);
    THCUNN_check_dim_size(state, gradOutput, 5, 3, outputHeight);
    THCUNN_check_dim_size(state, gradOutput, 5, 4, outputWidth);
  }
}


void THNN_(VolumetricUpSamplingNearest_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int outputDepth,
           int outputHeight,
           int outputWidth)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  int nbatch = THCTensor_(size)(state, input, 0);
  int channels = THCTensor_(size)(state, input, 1);
  int inputDepth = THCTensor_(size)(state, input, 2);
  int inputHeight = THCTensor_(size)(state, input, 3);
  int inputWidth  = THCTensor_(size)(state, input, 4);

  THNN_(VolumetricUpSamplingNearest_shapeCheck)(state, input, NULL, nbatch, channels,
		  inputDepth, inputHeight, inputWidth,
		  outputDepth, outputHeight, outputWidth);
  THAssert(inputDepth > 0 && inputHeight > 0 && inputWidth > 0 &&
		  outputDepth > 0 && outputHeight > 0 && outputWidth > 0);

  THCTensor_(resize5d)(state, output,
                       THCTensor_(size)(state, input, 0),
                       THCTensor_(size)(state, input, 1),
                       outputDepth,
                       outputHeight,
                       outputWidth);
  THCTensor_(zero)(state, output);

  THCDeviceTensor<scalar_t, 5> idata = toDeviceTensor<scalar_t, 5>(state, input);
  THCDeviceTensor<scalar_t, 5> odata = toDeviceTensor<scalar_t, 5>(state, output);

  const int num_kernels = outputDepth * outputHeight * outputWidth;
  const int num_threads = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  nearest_neighbor_5d_kernel<scalar_t, accreal> <<<THCCeilDiv(num_kernels, num_threads), num_threads,
	 0, stream>>>(num_kernels, idata, odata);
  THCudaCheck(cudaGetLastError());
}



void THNN_(VolumetricUpSamplingNearest_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int nbatch,
           int nchannels,
           int inputDepth,
           int inputHeight,
           int inputWidth,
           int outputDepth,
           int outputHeight,
           int outputWidth)
{
  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
  THNN_(VolumetricUpSamplingNearest_shapeCheck)(state, NULL, gradOutput, nbatch, nchannels,
		  inputDepth, inputHeight, inputWidth,
		  outputDepth, outputHeight, outputWidth);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCTensor_(resize5d)(state, gradInput, nbatch, nchannels, inputDepth, inputHeight, inputWidth);

  THCTensor_(zero)(state, gradInput);
  THCDeviceTensor<scalar_t, 5> data1 = toDeviceTensor<scalar_t, 5>(state, gradInput);
  THCDeviceTensor<scalar_t, 5> data2 = toDeviceTensor<scalar_t, 5>(state, gradOutput);
  const int num_kernels = outputDepth * outputHeight * outputWidth;
  const int num_threads = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  nearest_neighbor_5d_kernel_backward<scalar_t, accreal> <<<THCCeilDiv(num_kernels, num_threads),
	  num_threads, 0, stream>>>(num_kernels, data1, data2);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradOutput);
}

#endif
