#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/TemporalUpSamplingNearest.cu"
#else

#include "../common.h"

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
     THCUNN_argCheck(state, input->_dim() == 3, 2, input,
                     "3D input tensor expected but got: %s");
  }

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, 3, 0, nBatch);
    THCUNN_check_dim_size(state, gradOutput, 3, 1, nChannels);
    THCUNN_check_dim_size(state, gradOutput, 3, 2, outputWidth);
  }
}
	/*
                        (THCState *state,THCTensor *input, THCTensor *gradOutput,
                         int outputWidth) {
  THArgCheck(input != NULL, 2, "3D input tensor expected but got NULL");
  THCUNN_argCheck(state, input->_dim() == 2 || input->_dim() == 3, 2, input,
                  "2D or 3D input tensor expected but got: %s");
  if (input->_dim() == 2) {
    int nChannels    = THCTensor_(size)(state, input, 0);
    int inputWidth   = THCTensor_(size)(state, input, 1);
    if (gradOutput != NULL) {
      THCUNN_check_dim_size(state, gradOutput, 2, 0, nChannels);
      THCUNN_check_dim_size(state, gradOutput, 2, 1, outputWidth);
    }
  } else {
    int nBatch       = THCTensor_(size)(state, input, 0);
    int nChannels    = THCTensor_(size)(state, input, 1);
    int inputWidth   = THCTensor_(size)(state, input, 2);
    if (gradOutput != NULL) {
      THCUNN_check_dim_size(state, gradOutput, 3, 0, nBatch);
      THCUNN_check_dim_size(state, gradOutput, 3, 1, nChannels);
      THCUNN_check_dim_size(state, gradOutput, 3, 2, outputWidth);
    }
  }
}
*/

void THNN_(TemporalUpSamplingNearest_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int outputWidth,
	   bool align_corners)
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

  input = THCTensor_(newContiguous)(state, input);
  THCDeviceTensor<real, 3> idata = toDeviceTensor<real, 3>(state, input);
  THCDeviceTensor<real, 3> odata = toDeviceTensor<real, 3>(state, output);

  const int num_kernels = outputWidth;
  const int num_threads = THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  nearest_neighbor_interp2_kernel<real, accreal> <<<THCCeilDiv(num_kernels, num_threads), num_threads,
	 0, stream>>>(num_kernels, align_corners, idata, odata);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, input);
}


/*
  // This is for allocating output Tensor
  int64_t no_elements = 1;
  for(int i = 0; i < input->_dim() - 1; i++){
    no_elements *= input->size[i];
  }
  no_elements *= outputWidth;

  int d1;
  int d2;

  if (input->dim() == 2) {
    d1 = output->size[0];
    d2 = output->size[1];
  } else {
    d1 = output->size[1];
    d2 = output->size[2];
  }

  real *input_data = THCTensor_(data)(state, input);
  real *output_data = THCTensor_(data)(state, output);

  // cuda blocks & threads:
  int64_t nthreads = 256;
  // Max number of blocks: http://en.wikipedia.org/wiki/CUDA
  // 65535 for SM 2.x, 2^32 -1 for >= 3.0
  // TODO: When we move to SM 3.5 we should update this
  int64_t n_xblocks = min(max((int)ceil((float)no_elements / nthreads), 1), 65535);
  int64_t n_yblocks = (int64_t)ceil((float)no_elements / (float)(n_xblocks * nthreads));
  if (n_yblocks > 65535) {
    THError("Input size is too large!  aborting");
  }
  dim3 blocks(n_xblocks, n_yblocks);
  dim3 threads(nthreads);

  // kernel:
  upscale<<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data, no_elements, scale_factor, d1, d2);
  THCudaCheck(cudaGetLastError());

  // final cut:
  THCTensor_(free)(state, input);
}
*/

void THNN_(TemporalUpSamplingNearest_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int nbatch,
	   int nchannels,
	   int inputWidth,
	   int outputWidth,
	   bool align_corners)
{

  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
  THNN_(TemporalUpSamplingNearest_shapeCheck)(state, NULL, gradOutput, nbatch, nchannels, inputWidth, outputWidth);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCTensor_(resize3d)(state, gradInput, nbatch, nchannels, inputWidth);

  THCTensor_(zero)(state, gradInput);
  THCDeviceTensor<real, 3> data1 = toDeviceTensor<real, 3>(state, gradInput);
  THCDeviceTensor<real, 3> data2 = toDeviceTensor<real, 3>(state, gradOutput);
  const int num_kernels = outputWidth;
  const int num_threads = 
	  THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  nearest_neighbor_interp2_kernel_backward<real, accreal> <<<THCCeilDiv(num_kernels, num_threads),
	  num_threads, 0, stream>>>(num_kernels, align_corners, data1, data2);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradInput);
  THCTensor_(free)(state, gradOutput);
}

  /*
  real *gradInput_data = THCTensor_(data)(state, gradInput);
  real *gradOutput_data = THCTensor_(data)(state, gradOutput);

  int64_t no_elements = 1;
  for(int i = 0; i < gradInput->dim(); i++){
    no_elements *= gradInput->size[i];
  }

  int d1;
  int d2;

  if (gradInput->dim() == 2) {
    d1 = gradInput->size[0];
    d2 = gradInput->size[1];
  } else {
    d1 = gradInput->size[1];
    d2 = gradInput->size[2];
  }

  // cuda blocks & threads:
  int64_t nthreads = 256;
  // Max number of blocks: http://en.wikipedia.org/wiki/CUDA
  // 65535 for SM 2.x, 2^32 -1 for >= 3.0
  // TODO: When we move to SM 3.5 we should update this
  int64_t n_xblocks = min(max((int)ceil((float)no_elements / nthreads), 1), 65535);
  int64_t n_yblocks = (int64_t)ceil((float)no_elements / (float)(n_xblocks * nthreads));
  if (n_yblocks > 65535) {
    THError("Input size is too large!  aborting");
  }
  dim3 blocks(n_xblocks, n_yblocks);
  dim3 threads(nthreads);

  // kernel:
  downscale<real ,accreal> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data, no_elements,
    scale_factor, d1, d2);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradOutput);
}

*/
#endif
