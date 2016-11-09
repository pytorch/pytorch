#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/HardTanh.cu"
#else

#include "../common.h"

void THNN_(HardTanh_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           real min_val,
           real max_val,
           bool inplace)
{
  THCUNN_assertSameGPU_generic(state, 2, input, output);
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
           real min_val,
           real max_val,
           bool inplace)
{
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU_generic(state, 3, input, gradOutput, gradInput);

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
