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
  scalar_t margin = TH_CONVERT_ACCREAL_TO_REAL(margin_);
  THNN_CHECK_NELEMENT(input, target);
  THNN_CHECK_DIM_SIZE(output, 1, 0, 1);
  scalar_t sum = 0;

  TH_TENSOR_APPLY2(scalar_t, input, scalar_t, target,
    scalar_t z = (margin - *input_data * *target_data);
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
  scalar_t margin = TH_CONVERT_ACCREAL_TO_REAL(margin_);
  THNN_CHECK_NELEMENT(input, target);
  scalar_t norm = (sizeAverage ? 1./((scalar_t)THTensor_(nElement)(input)) : 1.);

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, input, scalar_t, target,
    *gradInput_data = (*input_data * *target_data) < margin ? -norm * *target_data : 0;
  );
}

#endif
