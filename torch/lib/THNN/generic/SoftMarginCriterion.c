#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftMarginCriterion.c"
#else

void THNN_(SoftMarginCriterion_updateOutput)(
  THNNState *state,
  THTensor *input,
  THTensor *target,
  THTensor *output,
  bool sizeAverage)
{
  THNN_CHECK_NELEMENT(input, target);
  THNN_CHECK_DIM_SIZE(output, 1, 0, 1);

  real sum;

  sum = 0;
  TH_TENSOR_APPLY2(real, input, real, target,
                   real z = log(1. + exp(-*input_data* *target_data));
                   sum += z;)

  if(sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(SoftMarginCriterion_updateGradInput)(
  THNNState *state,
  THTensor *input,
  THTensor *target,
  THTensor *gradInput,
  bool sizeAverage)
{
  THNN_CHECK_NELEMENT(input, target);
  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
                   real z = exp(-*target_data * *input_data);
                   *gradInput_data = -norm*(*target_data)*z/(1. + z);)
}

#endif
