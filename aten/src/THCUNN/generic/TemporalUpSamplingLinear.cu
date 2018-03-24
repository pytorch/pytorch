#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/TemporalUpSamplingLinear.cu"
#else

#include "../linear_upsampling.h"

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
     THCUNN_argCheck(state, input->nDimension == 3, 2, input,
                     "3D input tensor expected but got: %s");
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
  input = THCTensor_(newContiguous)(state, input);
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resize3d)(state, output,
                       THCTensor_(size)(state, input, 0),
                       THCTensor_(size)(state, input, 1),
                       outputWidth);
  THCTensor_(zero)(state, output);
  THCDeviceTensor<real, 3> idata = toDeviceTensor<real, 3>(state, input);
  THCDeviceTensor<real, 3> odata = toDeviceTensor<real, 3>(state, output);
  THAssert(inputWidth > 0 && outputWidth > 0);
  const accreal rwidth = linear_upsampling_compute_scale<accreal>(inputWidth, outputWidth, align_corners);
  const int num_kernels = outputWidth;
  const int num_threads =
    THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  caffe_gpu_interp2_kernel<real, accreal> <<<THCCeilDiv(num_kernels, num_threads), num_threads ,
   0 , stream>>>(num_kernels, rwidth, align_corners, idata, odata);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, input);
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
  gradInput = THCTensor_(newContiguous)(state, gradInput);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
  THCTensor_(resize3d)(state, gradInput, nbatch, nchannels, inputWidth);
  THCTensor_(zero)(state, gradInput);
  THCDeviceTensor<real, 3> data1 = toDeviceTensor<real, 3>(state, gradInput);
  THCDeviceTensor<real, 3> data2 = toDeviceTensor<real, 3>(state, gradOutput);
  const accreal rwidth = linear_upsampling_compute_scale<accreal>(inputWidth, outputWidth, align_corners);
  const int num_kernels = outputWidth;
  const int num_threads =
    THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  caffe_gpu_interp2_kernel_backward<real ,accreal> <<<THCCeilDiv(num_kernels, num_threads),
  num_threads, 0, stream>>>(num_kernels, rwidth, align_corners, data1, data2);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradInput);
  THCTensor_(free)(state, gradOutput);
}

#endif
