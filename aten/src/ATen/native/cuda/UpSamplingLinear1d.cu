// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou
#include <THCUNN/THCUNN.h>
#include <THC/THCTensor.hpp>
#include <THCUNN/common.h>
#include <THCUNN/upsampling.h>
#include <THC/THCDeviceTensor.cuh>
#include <THC/THCDeviceTensorUtils.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCAtomics.cuh>

template<typename Dtype, typename Acctype>
__global__ void caffe_gpu_interp2_kernel(const int n,
    const Acctype rwidth, const bool align_corners,
    const THCDeviceTensor<Dtype, 3> data1, THCDeviceTensor<Dtype, 3> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int width1 = data1.getSize(2);
  const int width2 = data2.getSize(2);

  if (index < n) {
    const int w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data1[n][c][w1];
          data2[n][c][w2] = val;
        }
      }
      return;
    }
    //
    const Acctype w1r = linear_upsampling_compute_source_index<Acctype>(rwidth, w2, align_corners);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = Acctype(1) - w1lambda;
    //
    for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
        const Acctype val = w0lambda * data1[n][c][w1]
                            + w1lambda * data1[n][c][w1+w1p];
        data2[n][c][w2] = ScalarConvert<Acctype, Dtype>::to(val);
      }
    }
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename Dtype, typename Acctype>
__global__ void caffe_gpu_interp2_kernel_backward(const int n,
    const Acctype rwidth, const bool align_corners,
    THCDeviceTensor<Dtype, 3> data1, const THCDeviceTensor<Dtype, 3> data2){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int width1 = data1.getSize(2);
  const int width2 = data2.getSize(2);
  if (index < n) {
    const int w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data2[n][c][w1];
          data1[n][c][w2] += val;
        }
      }
      return;
    }
    //
    const Acctype w1r = linear_upsampling_compute_source_index<Acctype>(rwidth, w2, align_corners);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = Acctype(1) - w1lambda;
    //
    for (int n = 0; n < batchsize ; n++){
      for (int c = 0; c < channels; ++c) {
        const Dtype d2val = data2[n][c][w2];
        atomicAdd(data1[n][c][w1].data(),
                  ScalarConvert<Acctype, Dtype>::to(w0lambda * d2val));
        atomicAdd(data1[n][c][w1+w1p].data(),
                  ScalarConvert<Acctype, Dtype>::to(w1lambda * d2val));
      }
    }
  }
}


#include <THCUNN/generic/TemporalUpSamplingLinear.cu>
#include <THC/THCGenerateFloatTypes.h>


// --------

#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/TemporalUpSamplingLinear.cu"
#else

#include <THCUNN/upsampling.h>
#include "ATen/cuda/CUDAContext.h"

static inline void THNN_(TemporalUpSamplingLinear_shapeCheck)
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
     THCUNN_argCheck(state, !input->is_empty() && input->dim() == 3, 2, input,
                     "non-empty 3D input tensor expected but got: %s");
  }

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, 3, 0, nBatch);
    THCUNN_check_dim_size(state, gradOutput, 3, 1, nChannels);
    THCUNN_check_dim_size(state, gradOutput, 3, 2, outputWidth);
  }
}

void THNN_(TemporalUpSamplingLinear_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int outputWidth,
           bool align_corners)
{
  int nbatch = THCTensor_(size)(state, input, 0);
  int channels = THCTensor_(size)(state, input, 1);
  int inputWidth = THCTensor_(size)(state, input, 2);
  THNN_(TemporalUpSamplingLinear_shapeCheck)
       (state, input, NULL,
        nbatch, channels,
        inputWidth, outputWidth);

  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resize3d)(state, output,
                       THCTensor_(size)(state, input, 0),
                       THCTensor_(size)(state, input, 1),
                       outputWidth);
  THCTensor_(zero)(state, output);
  THCDeviceTensor<scalar_t, 3> idata = toDeviceTensor<scalar_t, 3>(state, input);
  THCDeviceTensor<scalar_t, 3> odata = toDeviceTensor<scalar_t, 3>(state, output);
  THAssert(inputWidth > 0 && outputWidth > 0);
  const accreal rwidth = linear_upsampling_compute_scale<accreal>(inputWidth, outputWidth, align_corners);
  const int num_kernels = outputWidth;
  const int num_threads =
    at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  caffe_gpu_interp2_kernel<scalar_t, accreal> <<<THCCeilDiv(num_kernels, num_threads), num_threads ,
   0 , stream>>>(num_kernels, rwidth, align_corners, idata, odata);
  THCudaCheck(cudaGetLastError());
}


void THNN_(TemporalUpSamplingLinear_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int nbatch,
           int nchannels,
           int inputWidth,
           int outputWidth,
           bool align_corners)
{
  THNN_(TemporalUpSamplingLinear_shapeCheck)
       (state, NULL, gradOutput,
        nbatch, nchannels,
        inputWidth, outputWidth);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
  THCTensor_(resize3d)(state, gradInput, nbatch, nchannels, inputWidth);
  THCTensor_(zero)(state, gradInput);
  THCDeviceTensor<scalar_t, 3> data1 = toDeviceTensor<scalar_t, 3>(state, gradInput);
  THCDeviceTensor<scalar_t, 3> data2 = toDeviceTensor<scalar_t, 3>(state, gradOutput);
  const accreal rwidth = linear_upsampling_compute_scale<accreal>(inputWidth, outputWidth, align_corners);
  const int num_kernels = outputWidth;
  const int num_threads =
    at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  caffe_gpu_interp2_kernel_backward<scalar_t ,accreal> <<<THCCeilDiv(num_kernels, num_threads),
  num_threads, 0, stream>>>(num_kernels, rwidth, align_corners, data1, data2);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradOutput);
}

#endif
