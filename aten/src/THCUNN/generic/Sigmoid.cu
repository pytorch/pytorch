#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/Sigmoid.cu"
#else

#include <THCUNN/common.h>

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
  THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, gradInput, output, gradOutput, sigmoid_updateGradInput_functor<scalar_t>());
}

#endif
