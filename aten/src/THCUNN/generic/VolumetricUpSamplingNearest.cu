#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/VolumetricUpSamplingNearest.cu"
#else

#include <THCUNN/common.h>

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
  const int num_threads = THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
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
  const int num_threads = THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  nearest_neighbor_5d_kernel_backward<scalar_t, accreal> <<<THCCeilDiv(num_kernels, num_threads),
	  num_threads, 0, stream>>>(num_kernels, data1, data2);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradOutput);
}

#endif
