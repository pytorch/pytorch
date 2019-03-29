#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/HardTanh.cu"
#else

#include <THCUNN/common.h>

void THNN_(HardTanh_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal min_val_,
           accreal max_val_,
           bool inplace)
{
  scalar_t min_val = ScalarConvert<accreal, scalar_t>::to(min_val_);
  scalar_t max_val = ScalarConvert<accreal, scalar_t>::to(max_val_);

  THCUNN_assertSameGPU(state, 2, input, output);
  if(inplace)
  {
    THCTensor_(set)(state, output, input);
    THC_pointwiseApply1<scalar_t>(state, output, hardtanhupdateOutput_functor<scalar_t>(min_val, max_val));
  }
  else
  {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply2<scalar_t, scalar_t>(state, output, input,
                               hardtanhupdateOutput_functor<scalar_t>(min_val, max_val));
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
  scalar_t min_val = ScalarConvert<accreal, scalar_t>::to(min_val_);
  scalar_t max_val = ScalarConvert<accreal, scalar_t>::to(max_val_);

  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);

  if (inplace)
  {
    THCTensor_(set)(state, gradInput, gradOutput);
    THC_pointwiseApply2<scalar_t, scalar_t>(state, gradInput, input,
                                 hardtanhupdateGradInput_functor<scalar_t>(min_val, max_val));
  }
  else
  {
    THCTensor_(resizeAs)(state, gradInput, input);
    THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, gradInput, input, gradOutput,
                                 hardtanhupdateGradInput_functor<scalar_t>(min_val, max_val));
  }
}

#endif
