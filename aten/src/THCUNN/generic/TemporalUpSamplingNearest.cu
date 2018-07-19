#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/TemporalUpSamplingNearest.cu"
#else

#include "../common.h"

static inline void THNN_(TemporalUpSamplingNearest_shapeCheck)
                        (THCState *state,
                         THCTensor *input, THCTensor *gradOutput,
                         int64_t nBatch, int64_t nChannels,
                         int64_t inputWidth,
                         int64_t outputWidth) {
  THArgCheck(inputWidth > 0 && outputWidth > 0, 2,
             "input and output sizes should be greater than 0,"
             " but got input (W: %d) output (W: %d)",
             inputWidth, outputWidth);
  if (input != NULL) {
     THCUNN_argCheck(state, input->_dim() == 3, 2, input,
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
           int64_t outputWidth)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  int64_t nbatch = THCTensor_(size)(state, input, 0);
  int64_t channels = THCTensor_(size)(state, input, 1);
  int64_t inputWidth  = THCTensor_(size)(state, input, 2);

  THNN_(TemporalUpSamplingNearest_shapeCheck)(state, input, NULL, nbatch, channels, inputWidth, outputWidth);
  THAssert(inputWidth > 0 && outputWidth > 0);

  THCTensor_(resize3d)(state, output,
                       THCTensor_(size)(state, input, 0),
                       THCTensor_(size)(state, input, 1),
                       outputWidth);
  THCTensor_(zero)(state, output);

  THCDeviceTensor<real, 3> idata = toDeviceTensor<real, 3>(state, input);
  THCDeviceTensor<real, 3> odata = toDeviceTensor<real, 3>(state, output);

  const int64_t num_kernels = outputWidth;
  const int64_t num_threads = THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  nearest_neighbor_3d_kernel<real, accreal> <<<THCCeilDiv(num_kernels, num_threads), num_threads,
	 0, stream>>>(num_kernels, idata, odata);
  THCudaCheck(cudaGetLastError());
}


void THNN_(TemporalUpSamplingNearest_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int64_t nbatch,
           int64_t nchannels,
           int64_t inputWidth,
           int64_t outputWidth)
{
  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
  THNN_(TemporalUpSamplingNearest_shapeCheck)(state, NULL, gradOutput, nbatch, nchannels, inputWidth, outputWidth);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCTensor_(resize3d)(state, gradInput, nbatch, nchannels, inputWidth);

  THCTensor_(zero)(state, gradInput);
  THCDeviceTensor<real, 3> data1 = toDeviceTensor<real, 3>(state, gradInput);
  THCDeviceTensor<real, 3> data2 = toDeviceTensor<real, 3>(state, gradOutput);

  const int64_t num_kernels = outputWidth;
  const int64_t num_threads = THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);

  nearest_neighbor_3d_kernel_backward<real, accreal> <<<THCCeilDiv(num_kernels, num_threads),
	  num_threads, 0, stream>>>(num_kernels, data1, data2);

  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradOutput);
}

#endif
