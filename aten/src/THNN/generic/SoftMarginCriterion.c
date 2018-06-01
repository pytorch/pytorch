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

    TH_TENSOR_APPLY3(real, input, real, target, real, output,
                     *output_data = log(1. + exp(-*input_data * *target_data));)
    return;
  }

  THTensor_(resize1d)(output, 1);

  real sum;

  sum = 0;
  TH_TENSOR_APPLY2(real, input, real, target,
                   real z = log(1. + exp(-*input_data* *target_data));
                   sum += z;)

  if (reduction == Reduction::ElementwiseMean)
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

    TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
                     real z = exp(-*target_data * *input_data);
                     *gradInput_data = -*target_data * z/(1. + z);)
    THTensor_(cmul)(gradInput, gradInput, gradOutput);
    return;
  }

  real norm = (reduction == Reduction::ElementwiseMean ? 1./((real)THTensor_(nElement)(input)) : 1.);

  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
                   real z = exp(-*target_data * *input_data);
                   *gradInput_data = -norm*(*target_data)*z/(1. + z) * THTensor_(fastGet1d)(gradOutput, 0);)
}

#endif
