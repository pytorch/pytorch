#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MSECriterion.c"
#else

void THNN_(MSECriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          int64_t reduction)
{
  THNN_CHECK_SHAPE(input, target);

  if (reduction != Reduction::None) {
    THTensor_(resize1d)(output, 1);

    accreal sum = 0;

    TH_TENSOR_APPLY2(scalar_t, input, scalar_t, target,
      accreal z = (*input_data - *target_data);
      sum += z*z;
    );

    if (reduction == Reduction::Mean)
      sum /= THTensor_(nElement)(input);

    THTensor_(set1d)(output, 0, (scalar_t)sum);
    return;
  }

  THTensor_(resizeAs)(output, input);
  TH_TENSOR_APPLY3(scalar_t, input, scalar_t, target, scalar_t, output,
      scalar_t z = (*input_data - *target_data);
      *output_data = z*z;
  );
}

void THNN_(MSECriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          int64_t reduction)
{
  THNN_CHECK_SHAPE(input, target);
  THTensor_(resizeAs)(gradInput, input);

  if (reduction != Reduction::None) {
    THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);
    scalar_t norm = reduction == Reduction::Mean ? 2./((scalar_t)THTensor_(nElement)(input)) : 2.;
    norm *= THTensor_(get1d)(gradOutput, 0);
    TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, input, scalar_t, target,
      *gradInput_data = norm * (*input_data - *target_data);
    );
    return;
  }

  THNN_CHECK_SHAPE(input, gradOutput);
  TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, input, scalar_t, target,
    *gradInput_data = 2. * (*input_data - *target_data);
  );
  TH_TENSOR_APPLY2(scalar_t, gradInput, scalar_t, gradOutput,
    *gradInput_data *= *gradOutput_data;
  );
}

#endif
