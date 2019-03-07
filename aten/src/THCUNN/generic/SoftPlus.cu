#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/SoftPlus.cu"
#else

#include <THCUNN/common.h>

void THNN_(SoftPlus_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal beta_,
           accreal threshold_)
{
  scalar_t beta = ScalarConvert<accreal, scalar_t>::to(beta_);
  scalar_t threshold = ScalarConvert<accreal, scalar_t>::to(threshold_);
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resizeAs)(state, output, input);
  THC_pointwiseApply2<scalar_t, scalar_t>(state, output, input, softPlusupdateOutput_functor<scalar_t>(threshold, beta));
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
  scalar_t beta = ScalarConvert<accreal, scalar_t>::to(beta_);
  scalar_t threshold = ScalarConvert<accreal, scalar_t>::to(threshold_);
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 4, input, output, gradOutput, gradInput);
  THCTensor_(resizeAs)(state, gradInput, output);
  THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, gradInput, output, gradOutput, softPlusupdateGradInput_functor<scalar_t>(threshold, beta));
}

#endif
