#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialAdaptiveMaxPooling.cu"
#else

#include "../common.h"

// 4d tensor B x D x H x W

void THNN_(SpatialAdaptiveMaxPooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCIndexTensor *indices,
           int osizeW,
           int osizeH)
{
  THCUNN_assertSameGPU(state, 3, input, output, indices);

  THCIndex_t *indices_data;
  real *output_data;
  real *input_data;

  THCUNN_argCheck(state, input->nDimension == 3 || input->nDimension == 4, 2, input,
                  "3D or 4D (batch mode) tensor expected for input, but got: %s");

  if (input->nDimension == 3) {
    int64_t sizeD  = input->size[0];
    int64_t isizeH = input->size[1];
    int64_t isizeW = input->size[2];

    int64_t istrideD = input->stride[0];
    int64_t istrideH = input->stride[1];
    int64_t istrideW = input->stride[2];

    input_data = THCTensor_(data)(state, input);

    THCTensor_(resize3d)(state, output, sizeD, osizeH, osizeW);
    THCIndexTensor_(resize3d)(state, indices, sizeD, osizeH, osizeW);

    indices_data = THCIndexTensor_(data)(state, indices);
    output_data = THCTensor_(data)(state, output);

    // cuda blocks & threads:
    int blocksH = (int)(16L / sizeD);
    blocksH = blocksH < 1 ? 1 : blocksH;
    dim3 blocks(sizeD, blocksH);
    dim3 threads(32, 8);

    // run maxpool kernel
    adaptivemaxpool <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data,
                                   indices_data,
                                   isizeH, isizeW, osizeH, osizeW,
                                   istrideD, istrideH, istrideW);
    THCudaCheck(cudaGetLastError());

  } else {
    input = THCTensor_(newContiguous)(state, input);
    int64_t sizeB  = input->size[0];
    int64_t sizeD  = input->size[1];
    int64_t isizeH = input->size[2];
    int64_t isizeW = input->size[3];

    int64_t istrideD = input->stride[1];
    int64_t istrideH = input->stride[2];
    int64_t istrideW = input->stride[3];

    input_data = THCTensor_(data)(state, input);

    THCTensor_(resize4d)(state, output, sizeB, sizeD, osizeH, osizeW);
    THCIndexTensor_(resize4d)(state, indices, sizeB, sizeD, osizeH, osizeW);

    indices_data = THCIndexTensor_(data)(state, indices);
    output_data = THCTensor_(data)(state, output);

    // cuda blocks & threads:
    int blocksH = (int)(16L / sizeD);
    blocksH = blocksH < 1 ? 1 : blocksH;
    dim3 blocks(sizeB*sizeD, blocksH);
    dim3 threads(32, 8);

    // run maxpool kernel
    adaptivemaxpool <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data,
                                   indices_data,
                                   isizeH, isizeW, osizeH, osizeW,
                                   istrideD, istrideH, istrideW);
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
    int64_t sizeD  = input->size[0];
    int64_t isizeH = input->size[1];
    int64_t isizeW = input->size[2];

    int64_t osizeH = gradOutput->size[1];
    int64_t osizeW = gradOutput->size[2];

    //bool atomic = (isizeH%osizeH != 0) || (isizeW%osizeW != 0);

    THCTensor_(resizeAs)(state, gradInput, input);
    THCTensor_(zero)(state, gradInput);

    indices_data = THCIndexTensor_(data)(state, indices);
    gradOutput_data = THCTensor_(data)(state, gradOutput);
    gradInput_data = THCTensor_(data)(state, gradInput);

    // cuda blocks & threads:
    int blocksH = (int)(16L / sizeD);
    blocksH = blocksH < 1 ? 1 : blocksH;
    dim3 blocks(sizeD, blocksH);
    dim3 threads(32, 8);

    if(atomic)
    {
      // run updateGradInput kernel, accumulate gradients atomically
      atomicadaptivemaxgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          indices_data,
                                          isizeH, isizeW, osizeH, osizeW);
    }
    else
    {
      // run updateGradInput kernel
      atomicadaptivemaxgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          indices_data,
                                          isizeH, isizeW, osizeH, osizeW);
    }
    THCudaCheck(cudaGetLastError());
  } else {
    int64_t sizeB  = input->size[0];
    int64_t sizeD  = input->size[1];
    int64_t isizeH = input->size[2];
    int64_t isizeW = input->size[3];

    int64_t osizeH = gradOutput->size[2];
    int64_t osizeW = gradOutput->size[3];

    //bool atomic = (isizeH%osizeH != 0) || (isizeW%osizeW != 0);

    THCTensor_(resizeAs)(state, gradInput, input);
    THCTensor_(zero)(state, gradInput);

    indices_data = THCIndexTensor_(data)(state, indices);
    gradOutput_data = THCTensor_(data)(state, gradOutput);
    gradInput_data = THCTensor_(data)(state, gradInput);

    // cuda blocks & threads:
    int blocksH = (int)(16L / sizeD);
    blocksH = blocksH < 1 ? 1 : blocksH;
    dim3 blocks(sizeB*sizeD, blocksH);
    dim3 threads(32, 8);

    if(atomic)
    {
      // run updateGradInput kernel, accumulate gradients atomically
      atomicadaptivemaxgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          indices_data,
                                          isizeH, isizeW, osizeH, osizeW);
    }
    else
    {
      // run updateGradInput kernel, accumulate gradients atomically
      adaptivemaxgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          indices_data,
                                          isizeH, isizeW, osizeH, osizeW);
    }
    THCudaCheck(cudaGetLastError());
  }

  // clean
  THCTensor_(free)(state,gradOutput);

}

#endif
