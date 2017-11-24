#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Tanh.cu"
#else

#include "../common.h"

void THNN_(Tanh_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resizeAs)(state, output, input);
  THCTensor_(tanh)(state, output, input);
}

void THNN_(Tanh_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output)
{
  THCUNN_check_shape(state, output, gradOutput);
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);
  THCTensor_(resizeAs)(state, gradInput, output);
  THC_pointwiseApply3(state, gradInput, output, gradOutput, tanh_updateGradInput_functor<real>());
}

#endif
