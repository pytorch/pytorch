#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/DistKLDivCriterion.c"
#else

void THNN_(DistKLDivCriterion_updateOutput)(
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
      *output_data = *target_data > 0 ? *target_data * (log(*target_data) - *input_data) : 0;
    );
    return;
  }

  THTensor_(resize1d)(output, 1);

  scalar_t sum = 0;

  TH_TENSOR_APPLY2(scalar_t, input, scalar_t, target,
    sum += *target_data > 0 ? *target_data * (log(*target_data) - *input_data) : 0;
  );

  if (reduction == Reduction::ElementwiseMean)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(DistKLDivCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          int64_t reduction)
{
  THNN_CHECK_SHAPE(input, target);
  THTensor_(resizeAs)(gradInput, input);

  if (reduction == Reduction::None) {
    THNN_CHECK_SHAPE(input, gradOutput);
    TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, gradOutput, scalar_t, target,
      *gradInput_data = *target_data > 0 ? (-*target_data) * *gradOutput_data : 0;
    );
    return;
  }

  THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);

  scalar_t norm = (reduction == Reduction::ElementwiseMean ? 1./((scalar_t)THTensor_(nElement)(input)) : 1.);

  TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, input, scalar_t, target,
    *gradInput_data = *target_data > 0 ? norm * (-*target_data) * THTensor_(fastGetLegacy1dNoScalars)(gradOutput, 0) : 0;
  );
}

#endif
