#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Sqrt.cu"
#else

#include "../common.h"

void THNN_(Sqrt_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal eps_)
{
  real eps = ScalarConvert<accreal, real>::to(eps_);
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resizeAs)(state, output, input);
  THC_pointwiseApply2(state, output, input, sqrtupdateOutput_functor<real>(eps));
}

void THNN_(Sqrt_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output)
{
  THCUNN_check_shape(state, output, gradOutput);
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);
  THCTensor_(resizeAs)(state, gradInput, output);
  THC_pointwiseApply3(state, gradInput, output, gradOutput, sqrtupdateGradInput_functor<real>());
}

#endif
