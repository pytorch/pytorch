#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MarginCriterion.c"
#else

void THNN_(MarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          accreal margin_)
{
  real margin = TH_CONVERT_ACCREAL_TO_REAL(margin_);
  THNN_CHECK_NELEMENT(input, target);
  THNN_CHECK_DIM_SIZE(output, 1, 0, 1);
  real sum = 0;

  TH_TENSOR_APPLY2(real, input, real, target,
    real z = (margin - *input_data * *target_data);
    sum += z>0 ? z : 0;
  );

  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(MarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          accreal margin_)
{
  real margin = TH_CONVERT_ACCREAL_TO_REAL(margin_);
  THNN_CHECK_NELEMENT(input, target);
  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
    *gradInput_data = (*input_data * *target_data) < margin ? -norm * *target_data : 0;
  );
}

#endif
