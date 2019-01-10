#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricUpSamplingTrilinear.cu"
#else

#include "../linear_upsampling.h"

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
     THCUNN_argCheck(state, input->nDimension == 5, 2, input,
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
  input = THCTensor_(newContiguous)(state, input);
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resize5d)(state, output,
                       THCTensor_(size)(state, input, 0),
                       THCTensor_(size)(state, input, 1),
                       outputDepth, outputHeight, outputWidth);
  THCTensor_(zero)(state, output);
  THCDeviceTensor<real, 5> idata = toDeviceTensor<real, 5>(state, input);
  THCDeviceTensor<real, 5> odata = toDeviceTensor<real, 5>(state, output);
  THAssert(inputDepth > 0 && inputHeight > 0 && inputWidth > 0 && outputDepth > 0 && outputHeight > 0 && outputWidth > 0);
  const accreal rdepth = linear_upsampling_compute_scale<accreal>(inputDepth, outputDepth, align_corners);
  const accreal rheight = linear_upsampling_compute_scale<accreal>(inputHeight, outputHeight, align_corners);
  const accreal rwidth = linear_upsampling_compute_scale<accreal>(inputWidth, outputWidth, align_corners);
  const int num_kernels = outputDepth * outputHeight * outputWidth;
  const int num_threads =
    THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  caffe_gpu_interp2_kernel<real, accreal> <<<THCCeilDiv(num_kernels, num_threads), num_threads ,
   0 , stream>>>(num_kernels, rdepth, rheight, rwidth, align_corners, idata, odata);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, input);
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
  gradInput = THCTensor_(newContiguous)(state, gradInput);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
  THCTensor_(resize5d)(state, gradInput, nbatch, nchannels, inputDepth, inputHeight, inputWidth);
  THCTensor_(zero)(state, gradInput);
  THCDeviceTensor<real, 5> data1 = toDeviceTensor<real, 5>(state, gradInput);
  THCDeviceTensor<real, 5> data2 = toDeviceTensor<real, 5>(state, gradOutput);
  const accreal rdepth = linear_upsampling_compute_scale<accreal>(inputDepth, outputDepth, align_corners);
  const accreal rheight = linear_upsampling_compute_scale<accreal>(inputHeight, outputHeight, align_corners);
  const accreal rwidth = linear_upsampling_compute_scale<accreal>(inputWidth, outputWidth, align_corners);
  const int num_kernels = outputDepth * outputHeight * outputWidth;
  const int num_threads =
    THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  caffe_gpu_interp2_kernel_backward<real ,accreal> <<<THCCeilDiv(num_kernels, num_threads),
  num_threads, 0, stream>>>(num_kernels, rdepth, rheight, rwidth, align_corners, data1, data2);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradInput);
  THCTensor_(free)(state, gradOutput);
}

#endif
