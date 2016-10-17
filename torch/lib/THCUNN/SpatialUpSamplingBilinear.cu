// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou
#include "THCUNN.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"

__global__ void caffe_gpu_interp2_kernel(const int n,
    const float rheight, const float rwidth,
    const THCDeviceTensor<float, 4> data1, THCDeviceTensor<float, 4> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int height1 = data1.getSize(2);
  const int width1 = data1.getSize(3);
  const int height2 = data2.getSize(2);
  const int width2 = data2.getSize(3);

  if (index < n) {
    const int w2 = index % width2; // 0:width2-1
    const int h2 = index / width2; // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
          const float val = data1[n][c][h1][w1];
          data2[n][c][h2][w2] = val;
        }
      }
      return;
    }
    //
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = 1.0f - h1lambda;
    //
    const float w1r = rwidth * w2;
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const float w1lambda = w1r - w1;
    const float w0lambda = 1.0f - w1lambda;
    //
    for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
        const float val = h0lambda * (w0lambda * data1[n][c][h1][w1]
                            + w1lambda * data1[n][c][h1][w1+w1p])
                            + h1lambda * (w0lambda * data1[n][c][h1+h1p][w1]
                            + w1lambda * data1[n][c][h1+h1p][w1+w1p]);
        data2[n][c][h2][w2] = val;
      }
    }
  }
}

void THNN_CudaSpatialUpSamplingBilinear_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
	  int outputHeight,
          int outputWidth) {
  input = THCudaTensor_newContiguous(state, input);
  output = THCudaTensor_newContiguous(state, output);
  THCUNN_assertSameGPU(state, 2, input, output);
  THCudaTensor_zero(state, output);
  THCDeviceTensor<float, 4> idata = toDeviceTensor<float, 4>(state, input);
  THCDeviceTensor<float, 4> odata = toDeviceTensor<float, 4>(state, output);
  int height1 = idata.getSize(2);
  int width1 = idata.getSize(3);
  int height2 = odata.getSize(2);
  int width2 = odata.getSize(3);
  assert( height1 > 0 && width1 > 0 && height2 > 0 && width2 > 0);
  const float rheight= (height2 > 1) ? (float)(height1 - 1)/(height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? (float)(width1 - 1)/(width2 - 1) : 0.f;
  const int num_kernels = height2 * width2;
  const int num_threads =
    THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  caffe_gpu_interp2_kernel<<<THCCeilDiv(num_kernels, num_threads), num_threads ,
   0 , stream>>>(num_kernels, rheight, rwidth, idata, odata);
  THCudaCheck(cudaGetLastError());
  THCudaTensor_free(state, input);
  THCudaTensor_free(state, output);
}


// Backward (adjoint) operation 1 <- 2 (accumulates)
__global__ void caffe_gpu_interp2_kernel_backward(const int n,
    const float rheight, const float rwidth,
    THCDeviceTensor<float, 4> data1, const THCDeviceTensor<float, 4> data2){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int height1 = data1.getSize(2);
  const int width1 = data1.getSize(3);
  const int height2 = data2.getSize(2);
  const int width2 = data2.getSize(3);
  if (index < n) {
    const int w2 = index % width2; // 0:width2-1
    const int h2 = index / width2; // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
          const float val = data2[n][c][h1][w1];
          data1[n][c][h2][w2] += val;
        }
      }
      return;
    }
    //
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = 1.0f - h1lambda;
    //
    const float w1r = rwidth * w2;
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const float w1lambda = w1r - w1;
    const float w0lambda = 1.0f - w1lambda;
    //
    for (int n = 0; n < batchsize ; n++){
      for (int c = 0; c < channels; ++c) {
        const float d2val = data2[n][c][h2][w2];
        atomicAdd(data1[n][c][h1][w1].data(), h0lambda * w0lambda * d2val);
        atomicAdd(data1[n][c][h1][w1+w1p].data(), h0lambda * w1lambda * d2val);
        atomicAdd(data1[n][c][h1+h1p][w1].data(), h1lambda * w0lambda * d2val);
        atomicAdd(data1[n][c][h1+h1p][w1+w1p].data(),
                                                  h1lambda * w1lambda * d2val);
      }
    }
  }
}


void THNN_CudaSpatialUpSamplingBilinear_updateGradInput(
          THCState *state,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          int nbatch,
          int nchannels,
          int inputHeight,
          int inputWidth,
          int outputHeight,
          int outputWidth) {
  gradInput = THCudaTensor_newContiguous(state, gradInput);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
  THCudaTensor_zero(state, gradInput);
  THCDeviceTensor<float, 4> data1 = toDeviceTensor<float, 4>(state, gradInput);
  THCDeviceTensor<float, 4> data2 = toDeviceTensor<float, 4>(state, gradOutput);
  int height1 = data1.getSize(2);
  int width1 = data1.getSize(3);
  int height2 = data2.getSize(2);
  int width2 = data2.getSize(3);
  assert(height1 > 0 && width1 > 0 && height2 > 0 && width2 > 0);
  const float rheight= (height2 > 1) ? (float)(height1 - 1)/(height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? (float)(width1 - 1) / (width2 - 1) : 0.f;
  const int num_kernels = height2 * width2;
  const int num_threads =
    THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  caffe_gpu_interp2_kernel_backward<<<THCCeilDiv(num_kernels, num_threads),
  num_threads, 0, stream>>>(num_kernels, rheight, rwidth, data1, data2);
  THCudaCheck(cudaGetLastError());
  THCudaTensor_free(state, gradInput);
  THCudaTensor_free(state, gradOutput);
}
