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
__global__ void nearest_neighbor_3d_kernel(
		const int n,
		const THCDeviceTensor<Dtype, 3> data1,
		THCDeviceTensor<Dtype, 3> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int width1 = data1.getSize(2);
  const int width2 = data2.getSize(2);
  const float scale = (float) width1 / (float) width2;

  if (index < n) {
    const int w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int w1 = w2;
      for (int n = 0; n < batchsize; n++) {
	for (int c = 0; c < channels; ++c) {
	  const Dtype val = data1[n][c][w1];
	  data2[n][c][w2] = val;
	}
      }
      return;
    }
    //
    const int w1 = nearest_neighbor_compute_source_index(scale, w2, width1);
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
	const Dtype val = data1[n][c][w1];
	data2[n][c][w2] = val;
      }
    }
  }
}

// Backward operation
template <typename Dtype, typename Acctype>
__global__ void nearest_neighbor_3d_kernel_backward(
		const int n,
		THCDeviceTensor<Dtype, 3> data1,
		const THCDeviceTensor<Dtype, 3> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int width1 = data1.getSize(2);
  const int width2 = data2.getSize(2);
  const float scale = (float) width1 / (float) width2;

  if (index < n) {
    const int w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int w1 = w2;
      for (int n = 0; n < batchsize; n++) {
	for (int c = 0; c < channels; ++c) {
	  const Dtype val = data2[n][c][w1];
	  data1[n][c][w2] = val;
	}
      }
      return;
    }
    //
    const int w1 = nearest_neighbor_compute_source_index(scale, w2, width1);
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const Dtype d2val = data2[n][c][w2];
        atomicAdd(data1[n][c][w1].data(), d2val);
      }
    }
  }
}


#include <THCUNN/generic/TemporalUpSamplingNearest.cu>
#include <THC/THCGenerateFloatTypes.h>


// -------------------

#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/TemporalUpSamplingNearest.cu"
#else

#include <THCUNN/common.h>
#include "ATen/cuda/CUDAContext.h"

static inline void THNN_(TemporalUpSamplingNearest_shapeCheck)
                        (THCState *state,
                         THCTensor *input, THCTensor *gradOutput,
                         int nBatch, int nChannels,
                         int inputWidth,
                         int outputWidth) {
  THArgCheck(inputWidth > 0 && outputWidth > 0, 2,
             "input and output sizes should be greater than 0,"
             " but got input (W: %d) output (W: %d)",
             inputWidth, outputWidth);
  if (input != NULL) {
     THCUNN_argCheck(state, THTensor_nDimensionLegacyAll(input) == 3, 2, input,
                     "3D input tensor expected but got: %s");
  }

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, 3, 0, nBatch);
    THCUNN_check_dim_size(state, gradOutput, 3, 1, nChannels);
    THCUNN_check_dim_size(state, gradOutput, 3, 2, outputWidth);
  }
}

void THNN_(TemporalUpSamplingNearest_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int outputWidth)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  int nbatch = THCTensor_(size)(state, input, 0);
  int channels = THCTensor_(size)(state, input, 1);
  int inputWidth  = THCTensor_(size)(state, input, 2);

  THNN_(TemporalUpSamplingNearest_shapeCheck)(state, input, NULL, nbatch, channels, inputWidth, outputWidth);
  THAssert(inputWidth > 0 && outputWidth > 0);

  THCTensor_(resize3d)(state, output,
                       THCTensor_(size)(state, input, 0),
                       THCTensor_(size)(state, input, 1),
                       outputWidth);
  THCTensor_(zero)(state, output);

  THCDeviceTensor<scalar_t, 3> idata = toDeviceTensor<scalar_t, 3>(state, input);
  THCDeviceTensor<scalar_t, 3> odata = toDeviceTensor<scalar_t, 3>(state, output);

  const int num_kernels = outputWidth;
  const int num_threads = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  nearest_neighbor_3d_kernel<scalar_t, accreal> <<<THCCeilDiv(num_kernels, num_threads), num_threads,
	 0, stream>>>(num_kernels, idata, odata);
  THCudaCheck(cudaGetLastError());
}


void THNN_(TemporalUpSamplingNearest_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int nbatch,
           int nchannels,
           int inputWidth,
           int outputWidth)
{
  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
  THNN_(TemporalUpSamplingNearest_shapeCheck)(state, NULL, gradOutput, nbatch, nchannels, inputWidth, outputWidth);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCTensor_(resize3d)(state, gradInput, nbatch, nchannels, inputWidth);

  THCTensor_(zero)(state, gradInput);
  THCDeviceTensor<scalar_t, 3> data1 = toDeviceTensor<scalar_t, 3>(state, gradInput);
  THCDeviceTensor<scalar_t, 3> data2 = toDeviceTensor<scalar_t, 3>(state, gradOutput);

  const int num_kernels = outputWidth;
  const int num_threads = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);

  nearest_neighbor_3d_kernel_backward<scalar_t, accreal> <<<THCCeilDiv(num_kernels, num_threads),
	  num_threads, 0, stream>>>(num_kernels, data1, data2);

  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradOutput);
}

#endif
