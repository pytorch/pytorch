#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialAdaptiveAveragePooling.cu"
#else

#include "../common.h"

// 4d tensor B x D x H x W

void THNN_(SpatialAdaptiveAveragePooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int osizeW,
           int osizeH)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  real *output_data;
  real *input_data;

  THCUNN_argCheck(state, !input->is_empty() && (input->dim() == 3 || input->dim() == 4), 2, input,
                  "non-empty 3D or 4D (batch mode) tensor expected for input, but got: %s");

  if (input->dim() == 3) {
    int64_t sizeD  = THTensor_sizeLegacyNoScalars(input, 0);
    int64_t isizeH = THTensor_sizeLegacyNoScalars(input, 1);
    int64_t isizeW = THTensor_sizeLegacyNoScalars(input, 2);

    int64_t istrideD = THTensor_strideLegacyNoScalars(input, 0);
    int64_t istrideH = THTensor_strideLegacyNoScalars(input, 1);
    int64_t istrideW = THTensor_strideLegacyNoScalars(input, 2);

    input_data = THCTensor_(data)(state, input);

    THCTensor_(resize3d)(state, output, sizeD, osizeH, osizeW);

    output_data = THCTensor_(data)(state, output);

    // cuda blocks & threads:
    int blocksH = max((int)(16L / sizeD), 1);
    dim3 blocks(sizeD, blocksH);
    dim3 threads(32, 8);

    // run averagepool kernel
    adaptiveaveragepool <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data,
                                   isizeH, isizeW, osizeH, osizeW,
                                   istrideD, istrideH, istrideW);
    THCudaCheck(cudaGetLastError());

  } else {
    input = THCTensor_(newContiguous)(state, input);
    int64_t sizeB  = THTensor_sizeLegacyNoScalars(input, 0);
    int64_t sizeD  = THTensor_sizeLegacyNoScalars(input, 1);
    int64_t isizeH = THTensor_sizeLegacyNoScalars(input, 2);
    int64_t isizeW = THTensor_sizeLegacyNoScalars(input, 3);

    int64_t istrideD = THTensor_strideLegacyNoScalars(input, 1);
    int64_t istrideH = THTensor_strideLegacyNoScalars(input, 2);
    int64_t istrideW = THTensor_strideLegacyNoScalars(input, 3);

    input_data = THCTensor_(data)(state, input);

    THCTensor_(resize4d)(state, output, sizeB, sizeD, osizeH, osizeW);

    output_data = THCTensor_(data)(state, output);

    // cuda blocks & threads:
    int blocksH = max((int)(16L / sizeD), 1);
    dim3 blocks(sizeB * sizeD, blocksH);
    dim3 threads(32, 8);

    // run averagepool kernel
    adaptiveaveragepool <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data,
                                   isizeH, isizeW, osizeH, osizeW,
                                   istrideD, istrideH, istrideW);
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

  if (input->dim() == 3) {
    int64_t sizeD  = THTensor_sizeLegacyNoScalars(input, 0);
    int64_t isizeH = THTensor_sizeLegacyNoScalars(input, 1);
    int64_t isizeW = THTensor_sizeLegacyNoScalars(input, 2);

    int64_t osizeH = THTensor_sizeLegacyNoScalars(gradOutput, 1);
    int64_t osizeW = THTensor_sizeLegacyNoScalars(gradOutput, 2);

    //bool atomic = (isizeW%osizeW != 0) || (isizeH%osizeH != 0);

    THCTensor_(resizeAs)(state, gradInput, input);
    THCTensor_(zero)(state, gradInput);

    gradOutput_data = THCTensor_(data)(state, gradOutput);
    gradInput_data = THCTensor_(data)(state, gradInput);

    // cuda blocks & threads:
    int blocksH = max((int)(16L / sizeD), 1);
    dim3 blocks(sizeD, blocksH);
    dim3 threads(32, 8);

    if(atomic)
    {
      // run updateGradInput kernel, accumulate gradients atomically
      atomicadaptiveaveragegradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          isizeH, isizeW, osizeH, osizeW);
    }
    else
    {
      // run updateGradInput kernel
      adaptiveaveragegradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          isizeH, isizeW, osizeH, osizeW);
    }
    THCudaCheck(cudaGetLastError());
  } else {
    int64_t sizeB  = THTensor_sizeLegacyNoScalars(input, 0);
    int64_t sizeD  = THTensor_sizeLegacyNoScalars(input, 1);
    int64_t isizeH = THTensor_sizeLegacyNoScalars(input, 2);
    int64_t isizeW = THTensor_sizeLegacyNoScalars(input, 3);

    int64_t osizeH = THTensor_sizeLegacyNoScalars(gradOutput, 2);
    int64_t osizeW = THTensor_sizeLegacyNoScalars(gradOutput, 3);

    //bool atomic = //(isizeW%osizeW != 0) || (isizeH%osizeH != 0);

    THCTensor_(resizeAs)(state, gradInput, input);
    THCTensor_(zero)(state, gradInput);

    gradOutput_data = THCTensor_(data)(state, gradOutput);
    gradInput_data = THCTensor_(data)(state, gradInput);

    // cuda blocks & threads:
    int blocksH = max((int)(16L / sizeD), 1);
    dim3 blocks(sizeB * sizeD, blocksH);
    dim3 threads(32, 8);

    if(atomic)
    {
      // run updateGradInput kernel, accumulate gradients atomically
      atomicadaptiveaveragegradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          isizeH, isizeW, osizeH, osizeW);
    }
    else
    {
      // run updateGradInput kernel, accumulate gradients atomically
      adaptiveaveragegradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data,
                                          isizeH, isizeW, osizeH, osizeW);
    }
    THCudaCheck(cudaGetLastError());
  }

  // clean
  THCTensor_(free)(state,gradOutput);

}

#endif
