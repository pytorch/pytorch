#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialAdaptiveAveragePooling.cu"
#else

#include "../common.h"

void THNN_(SpatialAdaptiveAveragePooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int nOutputCols,
           int nOutputRows)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  real *output_data;
  real *input_data;

  THCUNN_argCheck(state, input->nDimension == 3 || input->nDimension == 4, 2, input,
                  "3D or 4D (batch mode) tensor expected for input, but got: %s");

  if (input->nDimension == 3) {
    int64_t nInputCols = input->size[2];
    int64_t nInputRows = input->size[1];
    int64_t nInputPlane = input->size[0];

    int64_t istride_d = input->stride[0];
    int64_t istride_h = input->stride[1];
    int64_t istride_w = input->stride[2];

    input_data = THCTensor_(data)(state, input);

    THCTensor_(resize3d)(state, output, nInputPlane, nOutputRows, nOutputCols);

    output_data = THCTensor_(data)(state, output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    // run averagepool kernel
    adaptiveaveragepool <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data,
                                   nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
                                   istride_h, istride_w, istride_d);
    THCudaCheck(cudaGetLastError());

  } else {
    input = THCTensor_(newContiguous)(state, input);
    int64_t nInputCols = input->size[3];
    int64_t nInputRows = input->size[2];
    int64_t nInputPlane = input->size[1];
    int64_t nbatch = input->size[0];

    int64_t istride_d = input->stride[1];
    int64_t istride_h = input->stride[2];
    int64_t istride_w = input->stride[3];

    input_data = THCTensor_(data)(state, input);

    THCTensor_(resize4d)(state, output, nbatch, nInputPlane, nOutputRows, nOutputCols);

    output_data = THCTensor_(data)(state, output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    // run averagepool kernel
    adaptiveaveragepool <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data,
                                   nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
                                   istride_h, istride_w, istride_d);
    THCudaCheck(cudaGetLastError());
    // clean
    THCTensor_(free)(state, input);
  }
}

void THNN_(SpatialAdaptiveAveragePooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput)
{
  bool atomic = true; // suboptimal, but without atomic it doesn't pass the tests

  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);

  real *gradInput_data;
  real *gradOutput_data;

  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  if (input->nDimension == 3) {
    int64_t nInputCols = input->size[2];
    int64_t nInputRows = input->size[1];
    int64_t nInputPlane = input->size[0];
    int64_t nOutputCols = gradOutput->size[2];
    int64_t nOutputRows = gradOutput->size[1];

    //bool atomic = (nInputCols%nOutputCols != 0) || (nInputRows%nOutputRows != 0);

    THCTensor_(resizeAs)(state, gradInput, input);
    THCTensor_(zero)(state, gradInput);

    gradOutput_data = THCTensor_(data)(state, gradOutput);
    gradInput_data = THCTensor_(data)(state, gradInput);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    if(atomic)
    {
      // run updateGradInput kernel, accumulate gradients atomically
      atomicadaptiveaveragegradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
    }
    else
    {
      // run updateGradInput kernel
      adaptiveaveragegradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
    }
    THCudaCheck(cudaGetLastError());
  } else {
    int64_t nInputCols = input->size[3];
    int64_t nInputRows = input->size[2];
    int64_t nInputPlane = input->size[1];
    int64_t nbatch = input->size[0];
    int64_t nOutputCols = gradOutput->size[3];
    int64_t nOutputRows = gradOutput->size[2];

    //bool atomic = //(nInputCols%nOutputCols != 0) || (nInputRows%nOutputRows != 0);

    THCTensor_(resizeAs)(state, gradInput, input);
    THCTensor_(zero)(state, gradInput);

    gradOutput_data = THCTensor_(data)(state, gradOutput);
    gradInput_data = THCTensor_(data)(state, gradInput);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    if(atomic)
    {
      // run updateGradInput kernel, accumulate gradients atomically
      atomicadaptiveaveragegradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
    }
    else
    {
      // run updateGradInput kernel, accumulate gradients atomically
      adaptiveaveragegradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
    }
    THCudaCheck(cudaGetLastError());
  }

  // clean
  THCTensor_(free)(state,gradOutput);

}

#endif
