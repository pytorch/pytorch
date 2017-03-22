#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/ELU.cu"
#else

#include "../common.h"


void THNN_(ELU_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal alpha_,
           bool inplace)
{
  real alpha = ScalarConvert<accreal, real>::to(alpha_);
  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    THC_pointwiseApply1(state, input, ELUupdateOutputIP_functor<real>(alpha));
    THCTensor_(set)(state, output, input);
  }
  else
  {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply2(state, output, input, ELUupdateOutput_functor<real>(alpha));
  }
}


void THNN_(ELU_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output,
           accreal alpha_,
           bool inplace)
{
  real alpha = ScalarConvert<accreal, real>::to(alpha_);
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);

  if (inplace)
  {
    THC_pointwiseApply2(state, gradOutput, output, ELUupdateGradInputIP_functor<real>(alpha));
    THCTensor_(set)(state, gradInput, gradOutput);
  }
  else
  {
    THCTensor_(resizeAs)(state, gradInput, output);
    THC_pointwiseApply3(state, gradInput, output, gradOutput, ELUupdateGradInput_functor<real>(alpha));
  }
}

#endif
