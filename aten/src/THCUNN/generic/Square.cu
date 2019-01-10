#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/Square.cu"
#else

#include <THCUNN/common.h>

void THNN_(Square_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resizeAs)(state, output, input);
  THC_pointwiseApply2<scalar_t, scalar_t>(state, output, input, squareupdateOutput_functor<scalar_t>());
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
  THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, gradInput, input, gradOutput, squareupdateGradInput_functor<scalar_t>());
}

#endif
