#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftMarginCriterion.c"
#else

void THNN_(SoftMarginCriterion_updateOutput)(
  THNNState *state,
  THTensor *input,
  THTensor *target,
  THTensor *output,
  int64_t reduction)
{
  THNN_CHECK_SHAPE(input, target);

  if (reduction == Reduction::None) {
    THTensor_(resizeAs)(output, input);

    TH_TENSOR_APPLY3(scalar_t, input, scalar_t, target, scalar_t, output,
                     *output_data = log(1. + exp(-*input_data * *target_data));)
    return;
  }

  THTensor_(resize1d)(output, 1);

  scalar_t sum;

  sum = 0;
  TH_TENSOR_APPLY2(scalar_t, input, scalar_t, target,
                   scalar_t z = log(1. + exp(-*input_data* *target_data));
                   sum += z;)

  if (reduction == Reduction::Mean)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(SoftMarginCriterion_updateGradInput)(
  THNNState *state,
  THTensor *input,
  THTensor *target,
  THTensor *gradOutput,
  THTensor *gradInput,
  int64_t reduction)
{
  THNN_CHECK_SHAPE(input, target);
  THTensor_(resizeAs)(gradInput, input);

  if (!reduction) {
    THNN_CHECK_SHAPE(gradOutput, input);

    TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, input, scalar_t, target,
                     scalar_t z = exp(-*target_data * *input_data);
                     *gradInput_data = -*target_data * z/(1. + z);)
    THTensor_(cmul)(gradInput, gradInput, gradOutput);
    return;
  }

  scalar_t norm = (reduction == Reduction::Mean ? 1./((scalar_t)THTensor_(nElement)(input)) : 1.);

  TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, input, scalar_t, target,
                   scalar_t z = exp(-*target_data * *input_data);
                   *gradInput_data = -norm*(*target_data)*z/(1. + z) * THTensor_(fastGetLegacy1dNoScalars)(gradOutput, 0);)
}

#endif
