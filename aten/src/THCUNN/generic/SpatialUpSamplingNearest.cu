#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialUpSamplingNearest.cu"
#else

#include "../common.h"

static inline void THNN_(SpatialUpSamplingNearest_shapeCheck)
                        (THCState *state,THCTensor *input, THCTensor *gradOutput,
                         int scale_factor) {
  THArgCheck(input != NULL, 2, "4D input tensor expected but got NULL");
  THArgCheck(scale_factor > 1, 4,
             "scale_factor must be greater than 1, but got: %d", scale_factor);
  THCUNN_argCheck(state, input->nDimension == 3 || input->nDimension == 4, 2, input,
                  "3D or 4D input tensor expected but got: %s");
  if (input->nDimension == 3) {
    int nChannels    = THCTensor_(size)(state, input, 0);
    int inputHeight  = THCTensor_(size)(state, input, 1);
    int inputWidth   = THCTensor_(size)(state, input, 2);
    int outputHeight = inputHeight * scale_factor;
    int outputWidth  = inputWidth  * scale_factor;
    if (gradOutput != NULL) {
      THCUNN_check_dim_size(state, gradOutput, 3, 0, nChannels);
      THCUNN_check_dim_size(state, gradOutput, 3, 1, outputHeight);
      THCUNN_check_dim_size(state, gradOutput, 3, 2, outputWidth);
    }
  } else {
    int nBatch       = THCTensor_(size)(state, input, 0);
    int nChannels    = THCTensor_(size)(state, input, 1);
    int inputHeight  = THCTensor_(size)(state, input, 2);
    int inputWidth   = THCTensor_(size)(state, input, 3);
    int outputHeight = inputHeight * scale_factor;
    int outputWidth  = inputWidth  * scale_factor;
    if (gradOutput != NULL) {
      THCUNN_check_dim_size(state, gradOutput, 4, 0, nBatch);
      THCUNN_check_dim_size(state, gradOutput, 4, 1, nChannels);
      THCUNN_check_dim_size(state, gradOutput, 4, 2, outputHeight);
      THCUNN_check_dim_size(state, gradOutput, 4, 3, outputWidth);
    }
  }
}

void THNN_(SpatialUpSamplingNearest_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int scale_factor)
{
  THCTensor_(zero)(state, output);

  THCUNN_assertSameGPU(state, 2, input, output);
  THNN_(SpatialUpSamplingNearest_shapeCheck)(state, input, NULL, scale_factor);
  int inputHeight = THCTensor_(size)(state, input, input->nDimension-2);
  int inputWidth  = THCTensor_(size)(state, input,  input->nDimension-1);
  int outputHeight = inputHeight * scale_factor;
  int outputWidth = inputWidth * scale_factor;

   if (input->nDimension == 3) {
     THCTensor_(resize3d)(state, output,
                          THCTensor_(size)(state, input, 0),
                          outputHeight, outputWidth);
   } else {
     THCTensor_(resize4d)(state, output,
                          THCTensor_(size)(state, input, 0),
                          THCTensor_(size)(state, input, 1),
                          outputHeight, outputWidth);
  }

  input = THCTensor_(newContiguous)(state, input);
  // This is for allocating output Tensor
  int64_t no_elements = 1;
  for(int i = 0; i < input->nDimension; i++){
    no_elements *= input->size[i];
  }
  no_elements *= scale_factor * scale_factor;

  int d1;
  int d2;
  int d3;

  if (input->nDimension == 3) {
    d1 = output->size[0];
    d2 = output->size[1];
    d3 = output->size[2];
  } else {
    d1 = output->size[1];
    d2 = output->size[2];
    d3 = output->size[3];
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
  upscale<<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data, no_elements, scale_factor, d1, d2, d3);
  THCudaCheck(cudaGetLastError());

  // final cut:
  THCTensor_(free)(state, input);
}

void THNN_(SpatialUpSamplingNearest_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int scale_factor)
{

  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
  THNN_(SpatialUpSamplingNearest_shapeCheck)(state, input, gradOutput, scale_factor);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCTensor_(resizeAs)(state, gradInput, input);

  THCTensor_(zero)(state, gradInput);

  real *gradInput_data = THCTensor_(data)(state, gradInput);
  real *gradOutput_data = THCTensor_(data)(state, gradOutput);

  int64_t no_elements = 1;
  for(int i = 0; i < gradInput->nDimension; i++){
    no_elements *= gradInput->size[i];
  }

  int d1;
  int d2;
  int d3;

  if (gradInput->nDimension == 3) {
    d1 = gradInput->size[0];
    d2 = gradInput->size[1];
    d3 = gradInput->size[2];
  } else {
    d1 = gradInput->size[1];
    d2 = gradInput->size[2];
    d3 = gradInput->size[3];
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
    scale_factor, d1, d2, d3);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradOutput);
}

#endif
