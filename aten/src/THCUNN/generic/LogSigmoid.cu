#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/LogSigmoid.cu"
#else

#include "../common.h"

void THNN_(LogSigmoid_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *buffer)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resizeAs)(state, output, input);
  THC_pointwiseApply2(state, output, input, logSigmoid_updateOutput_functor<real>());
}

void THNN_(LogSigmoid_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *buffer)
{
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);
  THCTensor_(resizeAs)(state, gradInput, input);
  THC_pointwiseApply3(state, gradInput, input, gradOutput, logSigmoid_updateGradInput_functor<real>());
}

#endif
