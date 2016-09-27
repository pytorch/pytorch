#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SmoothL1Criterion.c"
#else

void THNN_(SmoothL1Criterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage)
{
  THNN_CHECK_NELEMENT(input, target);
  THNN_CHECK_DIM_SIZE(output, 1, 0, 1);
  
  real sum = 0;
  TH_TENSOR_APPLY2(real, input, real, target,
    real z = fabs(*input_data - *target_data);
    sum += z < 1 ? 0.5*z*z : z - 0.5;
  );

  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(SmoothL1Criterion_updateGradInput)(
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
    real x = *input_data - *target_data;
    if (x < -1.)
     *gradInput_data = - norm;
    else if (x > 1.)
     *gradInput_data = norm;
    else
     *gradInput_data = norm * x;
  );
}

#endif
