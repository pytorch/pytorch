#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/ELU.cu"
#else

#include "../common.h"


void THNN_(ELU_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal alpha,
           accreal scale,
           bool inplace)
{
  real negcoef = ScalarConvert<accreal, real>::to(alpha * scale);
  real poscoef = ScalarConvert<accreal, real>::to(scale);
  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    THC_pointwiseApply1(state, input, ELUupdateOutputIP_functor<real>(negcoef, poscoef));
    THCTensor_(set)(state, output, input);
  }
  else
  {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply2(state, output, input, ELUupdateOutput_functor<real>(negcoef, poscoef));
  }
}


void THNN_(ELU_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output,
           accreal alpha,
           accreal scale)
{
  real negcoef = ScalarConvert<accreal, real>::to(alpha * scale);
  real poscoef = ScalarConvert<accreal, real>::to(scale);
  THCUNN_check_nElement(state, output, gradOutput);
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);

  THCTensor_(resizeAs)(state, gradInput, output);
  THC_pointwiseApply3(state, gradInput, output, gradOutput, ELUupdateGradInput_functor<real>(negcoef, poscoef));
}

#endif
