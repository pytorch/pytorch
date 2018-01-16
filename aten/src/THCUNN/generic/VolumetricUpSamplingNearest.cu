#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricUpSamplingNearest.cu"
#else

#include "../common.h"

static inline void THNN_(VolumetricUpSamplingNearest_shapeCheck)
                        (THCState *state,THCTensor *input, THCTensor *gradOutput,
                         int scale_factor) {
  THArgCheck(input != NULL, 2, "4D input tensor expected but got NULL");
  THArgCheck(scale_factor > 1, 4,
             "scale_factor must be greater than 1, but got: %d", scale_factor);
  THCUNN_argCheck(state, input->nDimension == 4 || input->nDimension == 5, 2, input,
                  "4D or 5D input tensor expected but got: %s");
  if (input->nDimension == 4) {
    int nChannels    = THCTensor_(size)(state, input, 0);
    int inputDepth   = THCTensor_(size)(state, input, 1);
    int inputHeight  = THCTensor_(size)(state, input, 2);
    int inputWidth   = THCTensor_(size)(state, input, 3);
    int outputDepth  = inputDepth * scale_factor;
    int outputHeight = inputHeight * scale_factor;
    int outputWidth  = inputWidth  * scale_factor;
    if (gradOutput != NULL) {
      THCUNN_check_dim_size(state, gradOutput, 4, 0, nChannels);
      THCUNN_check_dim_size(state, gradOutput, 4, 1, outputDepth);
      THCUNN_check_dim_size(state, gradOutput, 4, 2, outputHeight);
      THCUNN_check_dim_size(state, gradOutput, 4, 3, outputWidth);
    }
  } else {
    int nBatch       = THCTensor_(size)(state, input, 0);
    int nChannels    = THCTensor_(size)(state, input, 1);
    int inputDepth   = THCTensor_(size)(state, input, 2);
    int inputHeight  = THCTensor_(size)(state, input, 3);
    int inputWidth   = THCTensor_(size)(state, input, 4);
    int outputDepth  = inputDepth  * scale_factor;
    int outputHeight = inputHeight * scale_factor;
    int outputWidth  = inputWidth  * scale_factor;
    if (gradOutput != NULL) {
      THCUNN_check_dim_size(state, gradOutput, 5, 0, nBatch);
      THCUNN_check_dim_size(state, gradOutput, 5, 1, nChannels);
      THCUNN_check_dim_size(state, gradOutput, 5, 2, outputDepth);
      THCUNN_check_dim_size(state, gradOutput, 5, 3, outputHeight);
      THCUNN_check_dim_size(state, gradOutput, 5, 4, outputWidth);
    }
  }
}

void THNN_(VolumetricUpSamplingNearest_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int scale_factor)
{
  THCTensor_(zero)(state, output);

  THCUNN_assertSameGPU(state, 2, input, output);
  THNN_(VolumetricUpSamplingNearest_shapeCheck)(state, input, NULL, scale_factor);
  int inputDepth = THCTensor_(size)(state, input, input->nDimension-3);
  int inputHeight = THCTensor_(size)(state, input, input->nDimension-2);
  int inputWidth  = THCTensor_(size)(state, input,  input->nDimension-1);
  int outputDepth = inputDepth * scale_factor;
  int outputHeight = inputHeight * scale_factor;
  int outputWidth = inputWidth * scale_factor;

   if (input->nDimension == 4) {
     THCTensor_(resize4d)(state, output,
                          THCTensor_(size)(state, input, 0),
                          outputDepth, outputHeight, outputWidth);
   } else {
     THCTensor_(resize5d)(state, output,
                          THCTensor_(size)(state, input, 0),
                          THCTensor_(size)(state, input, 1),
                          outputDepth, outputHeight, outputWidth);
  }

  input = THCTensor_(newContiguous)(state, input);
  // This is for allocating output Tensor
  int64_t no_elements = 1;
  for(int i = 0; i < input->nDimension; i++){
    no_elements *= input->size[i];
  }
  no_elements *= scale_factor * scale_factor * scale_factor;

  int d1;
  int d2;
  int d3;
  int d4;

  if (input->nDimension == 4) {
    d1 = output->size[0];
    d2 = output->size[1];
    d3 = output->size[2];
    d4 = output->size[3];
  } else {
    d1 = output->size[1];
    d2 = output->size[2];
    d3 = output->size[3];
    d4 = output->size[4];
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
  vupscale<<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data, no_elements, scale_factor, d1, d2, d3, d4);
  THCudaCheck(cudaGetLastError());

  // final cut:
  THCTensor_(free)(state, input);
}

void THNN_(VolumetricUpSamplingNearest_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int scale_factor)
{

  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
  THNN_(VolumetricUpSamplingNearest_shapeCheck)(state, input, gradOutput, scale_factor);
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
  int d4;

  if (gradInput->nDimension == 4) {
    d1 = gradInput->size[0];
    d2 = gradInput->size[1];
    d3 = gradInput->size[2];
    d4 = gradInput->size[3];
  } else {
    d1 = gradInput->size[1];
    d2 = gradInput->size[2];
    d3 = gradInput->size[3];
    d4 = gradInput->size[4];
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
  vdownscale<real ,accreal> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data, no_elements,
    scale_factor, d1, d2, d3, d4);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradOutput);
}

#endif
