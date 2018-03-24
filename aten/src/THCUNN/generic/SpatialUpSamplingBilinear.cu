#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialUpSamplingBilinear.cu"
#else

#include "../linear_upsampling.h"

static inline void THNN_(SpatialUpSamplingBilinear_shapeCheck)
                        (THCState *state,
                         THCTensor *input, THCTensor *gradOutput,
                         int nBatch, int nChannels,
                         int inputHeight, int inputWidth,
                         int outputHeight, int outputWidth) {
  THArgCheck(inputHeight > 0 && inputWidth > 0
             && outputHeight > 0 && outputWidth > 0, 2,
             "input and output sizes should be greater than 0,"
             " but got input (H: %d, W: %d) output (H: %d, W: %d)",
             inputHeight, inputWidth, outputHeight, outputWidth);
  if (input != NULL) {
     THCUNN_argCheck(state, input->nDimension == 4, 2, input,
                     "4D input tensor expected but got: %s");
  }

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, 4, 0, nBatch);
    THCUNN_check_dim_size(state, gradOutput, 4, 1, nChannels);
    THCUNN_check_dim_size(state, gradOutput, 4, 2, outputHeight);
    THCUNN_check_dim_size(state, gradOutput, 4, 3, outputWidth);
  }
}

void THNN_(SpatialUpSamplingBilinear_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int outputHeight,
           int outputWidth,
           bool align_corners)
{
  int nbatch = THCTensor_(size)(state, input, 0);
  int channels = THCTensor_(size)(state, input, 1);
  int inputHeight = THCTensor_(size)(state, input, 2);
  int inputWidth = THCTensor_(size)(state, input, 3);
  THNN_(SpatialUpSamplingBilinear_shapeCheck)
       (state, input, NULL,
        nbatch, channels,
        inputHeight, inputWidth,
        outputHeight, outputWidth);
  input = THCTensor_(newContiguous)(state, input);
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resize4d)(state, output,
                       THCTensor_(size)(state, input, 0),
                       THCTensor_(size)(state, input, 1),
                       outputHeight, outputWidth);
  THCTensor_(zero)(state, output);
  THCDeviceTensor<real, 4> idata = toDeviceTensor<real, 4>(state, input);
  THCDeviceTensor<real, 4> odata = toDeviceTensor<real, 4>(state, output);
  THAssert(inputHeight > 0 && inputWidth > 0 && outputHeight > 0 && outputWidth > 0);
  const accreal rheight = linear_upsampling_compute_scale<accreal>(inputHeight, outputHeight, align_corners);
  const accreal rwidth = linear_upsampling_compute_scale<accreal>(inputWidth, outputWidth, align_corners);
  const int num_kernels = outputHeight * outputWidth;
  const int num_threads =
    THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  caffe_gpu_interp2_kernel<real, accreal> <<<THCCeilDiv(num_kernels, num_threads), num_threads ,
   0 , stream>>>(num_kernels, rheight, rwidth, align_corners, idata, odata);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, input);
}


void THNN_(SpatialUpSamplingBilinear_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int nbatch,
           int nchannels,
           int inputHeight,
           int inputWidth,
           int outputHeight,
           int outputWidth,
           bool align_corners)
{
  THNN_(SpatialUpSamplingBilinear_shapeCheck)
       (state, NULL, gradOutput,
        nbatch, nchannels,
        inputHeight, inputWidth,
        outputHeight, outputWidth);
  gradInput = THCTensor_(newContiguous)(state, gradInput);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
  THCTensor_(resize4d)(state, gradInput, nbatch, nchannels, inputHeight, inputWidth);
  THCTensor_(zero)(state, gradInput);
  THCDeviceTensor<real, 4> data1 = toDeviceTensor<real, 4>(state, gradInput);
  THCDeviceTensor<real, 4> data2 = toDeviceTensor<real, 4>(state, gradOutput);
  const accreal rheight = linear_upsampling_compute_scale<accreal>(inputHeight, outputHeight, align_corners);
  const accreal rwidth = linear_upsampling_compute_scale<accreal>(inputWidth, outputWidth, align_corners);
  const int num_kernels = outputHeight * outputWidth;
  const int num_threads =
    THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  caffe_gpu_interp2_kernel_backward<real ,accreal> <<<THCCeilDiv(num_kernels, num_threads),
  num_threads, 0, stream>>>(num_kernels, rheight, rwidth, align_corners, data1, data2);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradInput);
  THCTensor_(free)(state, gradOutput);
}

#endif
