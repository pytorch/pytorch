#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/ELU.cu"
#else

#include <THCUNN/common.h>


void THNN_(ELU_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal alpha,
           accreal scale,
           accreal input_scale,
           bool inplace)
{
  scalar_t negcoef = ScalarConvert<accreal, scalar_t>::to(alpha * scale);
  scalar_t poscoef = ScalarConvert<accreal, scalar_t>::to(scale);
  scalar_t negiptcoef = ScalarConvert<accreal, scalar_t>::to(input_scale);
  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    THC_pointwiseApply1<scalar_t>(state, input, ELUupdateOutputIP_functor<scalar_t>(negcoef, poscoef, negiptcoef));
    THCTensor_(set)(state, output, input);
  }
  else
  {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply2<scalar_t, scalar_t>(state, output, input, ELUupdateOutput_functor<scalar_t>(negcoef, poscoef, negiptcoef));
  }
}

void THNN_(ELU_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output,
           accreal alpha,
           accreal scale,
           accreal input_scale)
{
  scalar_t negcoef = ScalarConvert<accreal, scalar_t>::to(alpha * scale);
  scalar_t poscoef = ScalarConvert<accreal, scalar_t>::to(scale);
  scalar_t negiptcoef = ScalarConvert<accreal, scalar_t>::to(input_scale);
  THCUNN_check_nElement(state, output, gradOutput);
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);

  THCTensor_(resizeAs)(state, gradInput, output);
  THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, gradInput, output, gradOutput, ELUupdateGradInput_functor<scalar_t>(negcoef, poscoef, negiptcoef));
}

#endif
