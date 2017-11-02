#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialMaxPooling.cu"
#else

#include "../common.h"

void THNN_(SpatialMaxPooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCIndexTensor *indices,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           bool ceil_mode)
{
  THNN_(SpatialDilatedMaxPooling_updateOutput)(
    state, input, output, indices,
    kW, kH, dW, dH, padW, padH, 1, 1, ceil_mode);

}

void THNN_(SpatialMaxPooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCIndexTensor *indices,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           bool ceil_mode)
{
  THNN_(SpatialDilatedMaxPooling_updateGradInput)(
    state, input, gradOutput, gradInput, indices,
    kW, kH, dW, dH, padW, padH, 1, 1, ceil_mode);

}

#endif
