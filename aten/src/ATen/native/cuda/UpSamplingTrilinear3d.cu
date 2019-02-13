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
#include <c10/macros/Macros.h>

template<typename Dtype, typename Acctype>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void caffe_gpu_interp2_kernel(const int n,
    const Acctype rdepth, const Acctype rheight, const Acctype rwidth, const bool align_corners,
    const THCDeviceTensor<Dtype, 5> data1, THCDeviceTensor<Dtype, 5> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int depth1 = data1.getSize(2);
  const int height1 = data1.getSize(3);
  const int width1 = data1.getSize(4);
  const int depth2 = data2.getSize(2);
  const int height2 = data2.getSize(3);
  const int width2 = data2.getSize(4);

  if (index < n) {
    const int w2 = (index % (height2*width2)) % width2; // 0:width2-1
    const int h2 = (index % (height2*width2)) / width2; // 0:height2-1
    const int t2 = index / (height2*width2);            // 0:depth2-1
    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int t1 = t2;
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data1[n][c][t1][h1][w1];
          data2[n][c][t2][h2][w2] = val;
        }
      }
      return;
    }
    //
    const Acctype t1r = linear_upsampling_compute_source_index<Acctype>(rdepth, t2, align_corners);
    const int t1 = t1r;
    const int t1p = (t1 < depth1 - 1) ? 1 : 0;
    const Acctype t1lambda = t1r - t1;
    const Acctype t0lambda = Acctype(1) - t1lambda;
    //
    const Acctype h1r = linear_upsampling_compute_source_index<Acctype>(rheight, h2, align_corners);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Acctype h1lambda = h1r - h1;
    const Acctype h0lambda = Acctype(1) - h1lambda;
    //
    const Acctype w1r = linear_upsampling_compute_source_index<Acctype>(rwidth, w2, align_corners);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = Acctype(1) - w1lambda;
    //
    for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
        const Acctype val = t0lambda * (h0lambda * (w0lambda * data1[n][c][t1][h1][w1]
                                                  + w1lambda * data1[n][c][t1][h1][w1+w1p])
                                      + h1lambda * (w0lambda * data1[n][c][t1][h1+h1p][w1]
                                                  + w1lambda * data1[n][c][t1][h1+h1p][w1+w1p]))
                          + t1lambda * (h0lambda * (w0lambda * data1[n][c][t1+t1p][h1][w1]
                                                  + w1lambda * data1[n][c][t1+t1p][h1][w1+w1p])
                                      + h1lambda * (w0lambda * data1[n][c][t1+t1p][h1+h1p][w1]
                                                  + w1lambda * data1[n][c][t1+t1p][h1+h1p][w1+w1p]));
        data2[n][c][t2][h2][w2] = ScalarConvert<Acctype, Dtype>::to(val);
      }
    }
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename Dtype, typename Acctype>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void caffe_gpu_interp2_kernel_backward(const int n,
    const Acctype rdepth, const Acctype rheight, const Acctype rwidth, const bool align_corners,
    THCDeviceTensor<Dtype, 5> data1, const THCDeviceTensor<Dtype, 5> data2){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int depth1 = data1.getSize(2);
  const int height1 = data1.getSize(3);
  const int width1 = data1.getSize(4);
  const int depth2 = data2.getSize(2);
  const int height2 = data2.getSize(3);
  const int width2 = data2.getSize(4);
  if (index < n) {
    const int w2 = (index % (height2*width2)) % width2; // 0:width2-1
    const int h2 = (index % (height2*width2)) / width2; // 0:height2-1
    const int t2 = index / (height2*width2);            // 0:depth2-1
    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int t1 = t2;
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data2[n][c][t1][h1][w1];
          data1[n][c][t2][h2][w2] += val;
        }
      }
      return;
    }
    //
    const Acctype t1r = linear_upsampling_compute_source_index<Acctype>(rdepth, t2, align_corners);
    const int t1 = t1r;
    const int t1p = (t1 < depth1 - 1) ? 1 : 0;
    const Acctype t1lambda = t1r - t1;
    const Acctype t0lambda = Acctype(1) - t1lambda;
    //
    const Acctype h1r = linear_upsampling_compute_source_index<Acctype>(rheight, h2, align_corners);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Acctype h1lambda = h1r - h1;
    const Acctype h0lambda = Acctype(1) - h1lambda;
    //
    const Acctype w1r = linear_upsampling_compute_source_index<Acctype>(rwidth, w2, align_corners);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = Acctype(1) - w1lambda;
    //
    for (int n = 0; n < batchsize ; n++){
      for (int c = 0; c < channels; ++c) {
        const Dtype d2val = data2[n][c][t2][h2][w2];
        atomicAdd(data1[n][c][t1][h1][w1].data(),
                  ScalarConvert<Acctype, Dtype>::to(t0lambda * h0lambda * w0lambda * d2val));
        atomicAdd(data1[n][c][t1][h1][w1+w1p].data(),
                  ScalarConvert<Acctype, Dtype>::to(t0lambda * h0lambda * w1lambda * d2val));
        atomicAdd(data1[n][c][t1][h1+h1p][w1].data(),
                  ScalarConvert<Acctype, Dtype>::to(t0lambda * h1lambda * w0lambda * d2val));
        atomicAdd(data1[n][c][t1][h1+h1p][w1+w1p].data(),
                  ScalarConvert<Acctype, Dtype>::to(t0lambda * h1lambda * w1lambda * d2val));
        atomicAdd(data1[n][c][t1+t1p][h1][w1].data(),
                  ScalarConvert<Acctype, Dtype>::to(t1lambda * h0lambda * w0lambda * d2val));
        atomicAdd(data1[n][c][t1+t1p][h1][w1+w1p].data(),
                  ScalarConvert<Acctype, Dtype>::to(t1lambda * h0lambda * w1lambda * d2val));
        atomicAdd(data1[n][c][t1+t1p][h1+h1p][w1].data(),
                  ScalarConvert<Acctype, Dtype>::to(t1lambda * h1lambda * w0lambda * d2val));
        atomicAdd(data1[n][c][t1+t1p][h1+h1p][w1+w1p].data(),
                  ScalarConvert<Acctype, Dtype>::to(t1lambda * h1lambda * w1lambda * d2val));
      }
    }
  }
  /////////////////////////////////////////////////////////
}


#include <THCUNN/generic/VolumetricUpSamplingTrilinear.cu>
#include <THC/THCGenerateFloatTypes.h>


// --------

#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/VolumetricUpSamplingTrilinear.cu"
#else

#include <THCUNN/upsampling.h>
#include "ATen/cuda/CUDAContext.h"

static inline void THNN_(VolumetricUpSamplingTrilinear_shapeCheck)
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
     THCUNN_argCheck(state, !input->is_empty() && input->dim() == 5, 2, input,
                     "non-empty 5D input tensor expected but got: %s");
  }

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, 5, 0, nBatch);
    THCUNN_check_dim_size(state, gradOutput, 5, 1, nChannels);
    THCUNN_check_dim_size(state, gradOutput, 5, 2, outputDepth);
    THCUNN_check_dim_size(state, gradOutput, 5, 3, outputHeight);
    THCUNN_check_dim_size(state, gradOutput, 5, 4, outputWidth);
  }
}

void THNN_(VolumetricUpSamplingTrilinear_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int outputDepth,
           int outputHeight,
           int outputWidth,
           bool align_corners)
{
  int nbatch = THCTensor_(size)(state, input, 0);
  int channels = THCTensor_(size)(state, input, 1);
  int inputDepth = THCTensor_(size)(state, input, 2);
  int inputHeight = THCTensor_(size)(state, input, 3);
  int inputWidth = THCTensor_(size)(state, input, 4);
  THNN_(VolumetricUpSamplingTrilinear_shapeCheck)
       (state, input, NULL,
        nbatch, channels,
        inputDepth, inputHeight, inputWidth,
        outputDepth, outputHeight, outputWidth);

  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resize5d)(state, output,
                       THCTensor_(size)(state, input, 0),
                       THCTensor_(size)(state, input, 1),
                       outputDepth, outputHeight, outputWidth);
  THCTensor_(zero)(state, output);
  THCDeviceTensor<scalar_t, 5> idata = toDeviceTensor<scalar_t, 5>(state, input);
  THCDeviceTensor<scalar_t, 5> odata = toDeviceTensor<scalar_t, 5>(state, output);
  THAssert(inputDepth > 0 && inputHeight > 0 && inputWidth > 0 && outputDepth > 0 && outputHeight > 0 && outputWidth > 0);
  const accreal rdepth = linear_upsampling_compute_scale<accreal>(inputDepth, outputDepth, align_corners);
  const accreal rheight = linear_upsampling_compute_scale<accreal>(inputHeight, outputHeight, align_corners);
  const accreal rwidth = linear_upsampling_compute_scale<accreal>(inputWidth, outputWidth, align_corners);
  const int num_kernels = outputDepth * outputHeight * outputWidth;
  const int num_threads =
    at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  caffe_gpu_interp2_kernel<scalar_t, accreal> <<<THCCeilDiv(num_kernels, num_threads), num_threads ,
   0 , stream>>>(num_kernels, rdepth, rheight, rwidth, align_corners, idata, odata);
  THCudaCheck(cudaGetLastError());
}


void THNN_(VolumetricUpSamplingTrilinear_updateGradInput)(
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
           int outputWidth,
           bool align_corners)
{
  THNN_(VolumetricUpSamplingTrilinear_shapeCheck)
       (state, NULL, gradOutput,
        nbatch, nchannels,
        inputDepth, inputHeight, inputWidth,
        outputDepth, outputHeight, outputWidth);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
  THCTensor_(resize5d)(state, gradInput, nbatch, nchannels, inputDepth, inputHeight, inputWidth);
  THCTensor_(zero)(state, gradInput);
  THCDeviceTensor<scalar_t, 5> data1 = toDeviceTensor<scalar_t, 5>(state, gradInput);
  THCDeviceTensor<scalar_t, 5> data2 = toDeviceTensor<scalar_t, 5>(state, gradOutput);
  const accreal rdepth = linear_upsampling_compute_scale<accreal>(inputDepth, outputDepth, align_corners);
  const accreal rheight = linear_upsampling_compute_scale<accreal>(inputHeight, outputHeight, align_corners);
  const accreal rwidth = linear_upsampling_compute_scale<accreal>(inputWidth, outputWidth, align_corners);
  const int num_kernels = outputDepth * outputHeight * outputWidth;
  const int num_threads =
    at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  caffe_gpu_interp2_kernel_backward<scalar_t ,accreal> <<<THCCeilDiv(num_kernels, num_threads),
  num_threads, 0, stream>>>(num_kernels, rdepth, rheight, rwidth, align_corners, data1, data2);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradOutput);
}

#endif
