#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/BCECriterion.c"
#else

#define EPS 1e-12

void THNN_(BCECriterion_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *target,
    THTensor *output,
    bool sizeAverage,
    THTensor *weights,
    bool reduce)
{
  THNN_CHECK_NELEMENT(input, target);
  THNN_CHECK_NELEMENT(input, weights);

  if (!reduce) {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY3(real, input, real, target, real, output,
        real x = *input_data;
        real y = *target_data;
        THAssertMsg(x >= 0. && x <= 1.,
          "input value should be between 0~1, but got %f",
		      (double) x);
		    *output_data = -(log(x + EPS) * y + log(1. - x + EPS) * (1. - y));
    );
		if (weights) {
      THTensor_(cmul)(output, output, weights);
    }
    return;
  }

	THTensor_(resize1d)(output, 1);
  real sum = 0;

  if (weights) {
    TH_TENSOR_APPLY3(real, input, real, target, real, weights,
      real x = *input_data;
      real y = *target_data;
      real w = *weights_data;
      THAssertMsg(x >= 0. && x <= 1.,
        "input value should be between 0~1, but got %f",
		  (double) x);
      sum -= (log(x + EPS) * y + log(1. - x + EPS) * (1. - y)) * w;
    );
  } else {
    TH_TENSOR_APPLY2(real, input, real, target,
      real x = *input_data;
      real y = *target_data;
      THAssertMsg(x >= 0. && x <= 1.,
        "input value should be between 0~1, but got %f",
		  (double) x);
      sum -= log(x + EPS) * y + log(1. - x + EPS) * (1. - y);
    );
  }


  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(BCECriterion_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *target,
    THTensor *gradOutput,
    THTensor *gradInput,
    bool sizeAverage,
    THTensor *weights,
    bool reduce)
{
  THNN_CHECK_NELEMENT(input, target);
  THNN_CHECK_NELEMENT(input, weights);
  THTensor_(resizeAs)(gradInput, input);

  if (!reduce) {
    THNN_CHECK_NELEMENT(gradOutput, input);
    TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
      real x = *input_data;
      real y = *target_data;
      *gradInput_data = -(y - x) / ((1. - x + EPS) * (x + EPS));
    );

    if (weights) {
      TH_TENSOR_APPLY3(real, gradInput, real, weights, real, gradOutput,
        *gradInput_data = *gradInput_data * *weights_data * *gradOutput_data;
      );
    } else {
      THTensor_(cmul)(gradInput, gradInput, gradOutput);
    }
    return;
  }

  THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);
  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);

  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
    real x = *input_data;
    real y = *target_data;
    *gradInput_data = - norm * (y - x) / ((1. - x + EPS) * (x + EPS)) * THTensor_fastGet1d(gradOutput, 0);
  );

  if(weights)
    THTensor_(cmul)(gradInput, gradInput, weights);
}

#undef EPS

#endif
