#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Square.cu"
#else

#include "../common.h"

void THNN_(Square_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resizeAs)(state, output, input);
  THC_pointwiseApply2(state, output, input, squareupdateOutput_functor<real>());
}

void THNN_(Square_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput)
{
  THCUNN_check_shape(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);
  THCTensor_(resizeAs)(state, gradInput, input);
  THC_pointwiseApply3(state, gradInput, input, gradOutput, squareupdateGradInput_functor<real>());
}

#endif
