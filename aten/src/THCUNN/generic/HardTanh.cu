#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/HardTanh.cu"
#else

#include "../common.h"

void THNN_(HardTanh_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal min_val_,
           accreal max_val_,
           bool inplace)
{
  real min_val = ScalarConvert<accreal, real>::to(min_val_);
  real max_val = ScalarConvert<accreal, real>::to(max_val_);

  THCUNN_assertSameGPU(state, 2, input, output);
  if(inplace)
  {
    THCTensor_(set)(state, output, input);
    THC_pointwiseApply1(state, output, hardtanhupdateOutput_functor<real>(min_val, max_val));
  }
  else
  {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply2(state, output, input,
                               hardtanhupdateOutput_functor<real>(min_val, max_val));
  }
}

void THNN_(HardTanh_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           accreal min_val_,
           accreal max_val_,
           bool inplace)
{
  real min_val = ScalarConvert<accreal, real>::to(min_val_);
  real max_val = ScalarConvert<accreal, real>::to(max_val_);

  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);

  if (inplace)
  {
    THCTensor_(set)(state, gradInput, gradOutput);
    THC_pointwiseApply2(state, gradInput, input,
                                 hardtanhupdateGradInput_functor<real>(min_val, max_val));
  }
  else
  {
    THCTensor_(resizeAs)(state, gradInput, input);
    THC_pointwiseApply3(state, gradInput, input, gradOutput,
                                 hardtanhupdateGradInput_functor<real>(min_val, max_val));
  }
}

#endif
