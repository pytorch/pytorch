#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/AbsCriterion.c"
#else

void THNN_(AbsCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          int64_t reduction)
{
  THNN_CHECK_SHAPE(input, target);

  if (reduction == at::Reduction::None) {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY3(scalar_t, input, scalar_t, target, scalar_t, output,
      *output_data = fabs(*input_data - *target_data);
    );
    return;
  }

  scalar_t sum = 0;
  THTensor_(resize0d)(output);
  TH_TENSOR_APPLY2(scalar_t, input, scalar_t, target,
    sum += fabs(*input_data - *target_data);
  );

  if (reduction == at::Reduction::Mean)
    sum /= THTensor_(nElement)(input);

  THTensor_(set0d)(output, sum);
}

void THNN_(AbsCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          int64_t reduction)
{
  THNN_CHECK_SHAPE(input, target);
  THTensor_(resizeAs)(gradInput, input);

  if (reduction == at::Reduction::None) {
    THNN_CHECK_SHAPE(gradOutput, input);
    TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, input, scalar_t, target,
      *gradInput_data = ((*input_data - *target_data) >= 0 ? 1 : -1);
    );
    TH_TENSOR_APPLY2(scalar_t, gradInput, scalar_t, gradOutput,
      *gradInput_data *= *gradOutput_data;
    );
    return;
  }

  THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);
  scalar_t norm = (reduction == at::Reduction::Mean ? 1./((scalar_t)THTensor_(nElement)(input)) : 1.) * THTensor_(fastGetLegacy1dNoScalars)(gradOutput, 0);

  TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, input, scalar_t, target,
    *gradInput_data = (*input_data - *target_data) >= 0 ? norm : -norm;
  );
}

#endif
