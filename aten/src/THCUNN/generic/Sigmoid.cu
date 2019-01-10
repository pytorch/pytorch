#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Sigmoid.cu"
#else

#include "../common.h"

void THNN_(Sigmoid_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(sigmoid)(state, output, input);
}

void THNN_(Sigmoid_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output)
{
  THCUNN_check_nElement(state, output, gradOutput);
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);
  THCTensor_(resizeAs)(state, gradInput, output);
  THC_pointwiseApply3(state, gradInput, output, gradOutput, sigmoid_updateGradInput_functor<real>());
}

#endif
