#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Abs.cu"
#else

#include "../common.h"

void THNN_(Abs_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resizeAs)(state, output, input);
  THC_pointwiseApply2(state, output, input, absupdateOutput_functor<real>());
}

void THNN_(Abs_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput)
{
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);
  THCTensor_(resizeAs)(state, gradInput, input);
  THC_pointwiseApply3(state, gradInput, input, gradOutput, absupdateGradInput_functor<real>());
}

#endif
