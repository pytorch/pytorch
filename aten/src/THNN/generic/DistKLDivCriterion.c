#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/DistKLDivCriterion.c"
#else

void THNN_(DistKLDivCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          bool reduce)
{
  THNN_CHECK_SHAPE(input, target);

  if (!reduce) {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY3(real, input, real, target, real, output,
      *output_data = *target_data > 0 ? *target_data * (log(*target_data) - *input_data) : 0;
    );
    return;
  }

  THTensor_(resize1d)(output, 1);

  real sum = 0;

  TH_TENSOR_APPLY2(real, input, real, target,
    sum += *target_data > 0 ? *target_data * (log(*target_data) - *input_data) : 0;
  );

  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(DistKLDivCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          bool sizeAverage,
          bool reduce)
{
  THNN_CHECK_SHAPE(input, target);
  THTensor_(resizeAs)(gradInput, input);

  if (!reduce) {
    THNN_CHECK_SHAPE(input, gradOutput);
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, target,
      *gradInput_data = *target_data > 0 ? (-*target_data) * *gradOutput_data : 0;
    );
    return;
  }

  THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);

  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);

  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
    *gradInput_data = *target_data > 0 ? norm * (-*target_data) * THTensor_fastGet1d(gradOutput, 0) : 0;
  );
}

#endif
