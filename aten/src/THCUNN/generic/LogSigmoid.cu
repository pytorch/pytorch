#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/LogSigmoid.cu"
#else

#include <THCUNN/common.h>

void THNN_(LogSigmoid_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *buffer)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resizeAs)(state, output, input);
  THC_pointwiseApply2<scalar_t, scalar_t>(state, output, input, logSigmoid_updateOutput_functor<scalar_t>());
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
  THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, gradInput, input, gradOutput, logSigmoid_updateGradInput_functor<scalar_t>());
}

#endif
