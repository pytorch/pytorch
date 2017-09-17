#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialAdaptiveMaxPooling.cu"
#else

#include "../common.h"

void THNN_(SpatialAdaptiveMaxPooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCIndexTensor *indices,
           int nOutputCols,
           int nOutputRows)
{
  THCUNN_assertSameGPU(state, 3, input, output, indices);

  THCIndex_t *indices_data;
  real *output_data;
  real *input_data;

  THCUNN_argCheck(state, input->nDimension == 3 || input->nDimension == 4, 2, input,
                  "3D or 4D (batch mode) tensor expected for input, but got: %s");

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nInputPlane = input->size[0];

    long istride_d = input->stride[0];
    long istride_h = input->stride[1];
    long istride_w = input->stride[2];

    input_data = THCTensor_(data)(state, input);

    THCTensor_(resize3d)(state, output, nInputPlane, nOutputRows, nOutputCols);
    THCIndexTensor_(resize3d)(state, indices, nInputPlane, nOutputRows, nOutputCols);

    indices_data = THCIndexTensor_(data)(state, indices);
    output_data = THCTensor_(data)(state, output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    // run maxpool kernel
    adaptivemaxpool <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data,
                                   indices_data,
                                   nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
                                   istride_h, istride_w, istride_d);
    THCudaCheck(cudaGetLastError());

  } else {
    input = THCTensor_(newContiguous)(state, input);
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputPlane = input->size[1];
    long nbatch = input->size[0];

    long istride_d = input->stride[1];
    long istride_h = input->stride[2];
    long istride_w = input->stride[3];

    input_data = THCTensor_(data)(state, input);

    THCTensor_(resize4d)(state, output, nbatch, nInputPlane, nOutputRows, nOutputCols);
    THCIndexTensor_(resize4d)(state, indices, nbatch, nInputPlane, nOutputRows, nOutputCols);

    indices_data = THCIndexTensor_(data)(state, indices);
    output_data = THCTensor_(data)(state, output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    // run maxpool kernel
    adaptivemaxpool <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data,
                                   indices_data,
                                   nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
                                   istride_h, istride_w, istride_d);
    THCudaCheck(cudaGetLastError());
    // clean
    THCTensor_(free)(state, input);
  }
}

void THNN_(SpatialAdaptiveMaxPooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCIndexTensor *indices)
{
  bool atomic = true; // suboptimal, but without atomic it doesn't pass the tests

  THCUNN_assertSameGPU(state, 4, input, indices, gradOutput, gradInput);

  THCIndex_t *indices_data;
  real *gradInput_data;
  real *gradOutput_data;

  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nInputPlane = input->size[0];
    long nOutputCols = gradOutput->size[2];
    long nOutputRows = gradOutput->size[1];

    //bool atomic = (nInputCols%nOutputCols != 0) || (nInputRows%nOutputRows != 0);

    THCTensor_(resizeAs)(state, gradInput, input);
    THCTensor_(zero)(state, gradInput);

    indices_data = THCIndexTensor_(data)(state, indices);
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
      atomicadaptivemaxgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          indices_data,
                                          nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
    }
    else
    {
      // run updateGradInput kernel
      atomicadaptivemaxgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          indices_data,
                                          nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
    }
    THCudaCheck(cudaGetLastError());
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputPlane = input->size[1];
    long nbatch = input->size[0];
    long nOutputCols = gradOutput->size[3];
    long nOutputRows = gradOutput->size[2];

    //bool atomic = //(nInputCols%nOutputCols != 0) || (nInputRows%nOutputRows != 0);

    THCTensor_(resizeAs)(state, gradInput, input);
    THCTensor_(zero)(state, gradInput);

    indices_data = THCIndexTensor_(data)(state, indices);
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
      atomicadaptivemaxgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          indices_data,
                                          nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
    }
    else
    {
      // run updateGradInput kernel, accumulate gradients atomically
      adaptivemaxgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          indices_data,
                                          nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
    }
    THCudaCheck(cudaGetLastError());
  }

  // clean
  THCTensor_(free)(state,gradOutput);

}

#endif
