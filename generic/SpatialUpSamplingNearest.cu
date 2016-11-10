#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialUpSamplingNearest.cu"
#else

#include "../common.h"

void THNN_(SpatialUpSamplingNearest_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int scale_factor)
{
  // TODO: check argument shapes
  THCTensor_(zero)(state, output);

  THCUNN_assertSameGPU(state, 2, input, output);

  input = THCTensor_(newContiguous)(state, input);
  // This is for allocating output Tensor
  long no_elements = 1;
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
  long nthreads = 256;
  // Max number of blocks: http://en.wikipedia.org/wiki/CUDA
  // 65535 for SM 2.x, 2^32 -1 for >= 3.0
  // TODO: When we move to SM 3.5 we should update this
  long n_xblocks = min(max((int)ceil((float)no_elements / nthreads), 1), 65535);
  long n_yblocks = (long)ceil((float)no_elements / (float)(n_xblocks * nthreads));
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
  // TODO: check argument shapes
  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);

  THCTensor_(zero)(state, gradInput);

  real *gradInput_data = THCTensor_(data)(state, gradInput);
  real *gradOutput_data = THCTensor_(data)(state, gradOutput);

  long no_elements = 1;
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
  long nthreads = 256;
  // Max number of blocks: http://en.wikipedia.org/wiki/CUDA
  // 65535 for SM 2.x, 2^32 -1 for >= 3.0
  // TODO: When we move to SM 3.5 we should update this
  long n_xblocks = min(max((int)ceil((float)no_elements / nthreads), 1), 65535);
  long n_yblocks = (long)ceil((float)no_elements / (float)(n_xblocks * nthreads));
  if (n_yblocks > 65535) {
    THError("Input size is too large!  aborting");
  }
  dim3 blocks(n_xblocks, n_yblocks);
  dim3 threads(nthreads);

  // kernel:
  downscale<real ,accreal> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data, no_elements,
    scale_factor, d1, d2, d3);
  THCudaCheck(cudaGetLastError());
}

#endif
