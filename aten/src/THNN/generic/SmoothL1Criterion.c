#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SmoothL1Criterion.c"
#else

void THNN_(SmoothL1Criterion_updateOutput)(
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
      scalar_t z = fabs(*input_data - *target_data);
      *output_data = z < 1 ? 0.5 * z * z : z - 0.5;
    );
    return;
  }

  THTensor_(resize1d)(output, 1);

  scalar_t sum = 0;
  TH_TENSOR_APPLY2(scalar_t, input, scalar_t, target,
    scalar_t z = fabs(*input_data - *target_data);
    sum += z < 1 ? 0.5*z*z : z - 0.5;
  );

  if (reduction == Reduction::Mean)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(SmoothL1Criterion_updateGradInput)(
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
    THNN_CHECK_SHAPE(gradOutput, input);
    TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, input, scalar_t, target,
      scalar_t x = *input_data - *target_data;
      if (x < -1.) {
        *gradInput_data = -1.;
      } else if (x > 1.) {
        *gradInput_data = 1.;
      } else {
        *gradInput_data = x;
      }
    );
    TH_TENSOR_APPLY2(scalar_t, gradInput, scalar_t, gradOutput,
      *gradInput_data *= *gradOutput_data;
    );
    return;
  }

  THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);
  scalar_t norm = (reduction == Reduction::Mean ? 1./((scalar_t)THTensor_(nElement)(input)) : 1.) * THTensor_(fastGetLegacy1dNoScalars)(gradOutput, 0);

  TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, input, scalar_t, target,
    scalar_t x = *input_data - *target_data;
    if (x < -1.)
     *gradInput_data = - norm;
    else if (x > 1.)
     *gradInput_data = norm;
    else
     *gradInput_data = norm * x;
  );
}

#endif
