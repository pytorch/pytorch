#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SoftPlus.cu"
#else

#include "../common.h"

void THNN_(SoftPlus_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal beta_,
           accreal threshold_)
{
  real beta = ScalarConvert<accreal, real>::to(beta_);
  real threshold = ScalarConvert<accreal, real>::to(threshold_);
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resizeAs)(state, output, input);
  THC_pointwiseApply2(state, output, input, softPlusupdateOutput_functor<real>(threshold, beta));
}

void THNN_(SoftPlus_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output,
           accreal beta_,
           accreal threshold_)
{
  real beta = ScalarConvert<accreal, real>::to(beta_);
  real threshold = ScalarConvert<accreal, real>::to(threshold_);
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 4, input, output, gradOutput, gradInput);
  THCTensor_(resizeAs)(state, gradInput, output);
  THC_pointwiseApply3(state, gradInput, output, gradOutput, softPlusupdateGradInput_functor<real>(threshold, beta));
}

#endif
