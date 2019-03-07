#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/Abs.cu"
#else

#include <THCUNN/common.h>

void THNN_(Abs_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resizeAs)(state, output, input);
  THC_pointwiseApply2<scalar_t, scalar_t>(state, output, input, absupdateOutput_functor<scalar_t>());
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
  THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, gradInput, input, gradOutput, absupdateGradInput_functor<scalar_t>());
}

#endif
