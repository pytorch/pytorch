#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/BCECriterion.c"
#else

#define EPS 1e-12

static inline scalar_t safe_log(scalar_t a) {
  if (a == 0.) {
    return log(EPS);
  }
  return log(a);
}

void THNN_(BCECriterion_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *target,
    THTensor *output,
    int64_t reduction,
    THTensor *weights)
{
  THNN_CHECK_NELEMENT(input, target);
  THNN_CHECK_NELEMENT(input, weights);

  if (reduction == Reduction::None) {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY3(scalar_t, input, scalar_t, target, scalar_t, output,
        scalar_t x = *input_data;
        scalar_t y = *target_data;
        THAssertMsg(x >= 0. && x <= 1.,
          "input value should be between 0~1, but got %f",
		      (double) x);
		    *output_data = -(safe_log(x) * y + safe_log(1. - x) * (1. - y));
    );
		if (weights) {
      THTensor_(cmul)(output, output, weights);
    }
    return;
  }

	THTensor_(resize1d)(output, 1);
  scalar_t sum = 0;

  if (weights) {
    TH_TENSOR_APPLY3(scalar_t, input, scalar_t, target, scalar_t, weights,
      scalar_t x = *input_data;
      scalar_t y = *target_data;
      scalar_t w = *weights_data;
      THAssertMsg(x >= 0. && x <= 1.,
        "input value should be between 0~1, but got %f",
		  (double) x);
      sum -= (safe_log(x) * y + safe_log(1. - x) * (1. - y)) * w;
    );
  } else {
    TH_TENSOR_APPLY2(scalar_t, input, scalar_t, target,
      scalar_t x = *input_data;
      scalar_t y = *target_data;
      THAssertMsg(x >= 0. && x <= 1.,
        "input value should be between 0~1, but got %f",
		  (double) x);
      sum -= safe_log(x) * y + safe_log(1. - x) * (1. - y);
    );
  }


  if (reduction == Reduction::Mean)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(BCECriterion_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *target,
    THTensor *gradOutput,
    THTensor *gradInput,
    int64_t reduction,
    THTensor *weights)
{
  THNN_CHECK_NELEMENT(input, target);
  THNN_CHECK_NELEMENT(input, weights);
  THTensor_(resizeAs)(gradInput, input);

  if (reduction == Reduction::None) {
    THNN_CHECK_NELEMENT(gradOutput, input);
    TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, input, scalar_t, target,
      scalar_t x = *input_data;
      scalar_t y = *target_data;
      *gradInput_data = -(y - x) / ((1. - x + EPS) * (x + EPS));
    );

    if (weights) {
      TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, weights, scalar_t, gradOutput,
        *gradInput_data = *gradInput_data * *weights_data * *gradOutput_data;
      );
    } else {
      THTensor_(cmul)(gradInput, gradInput, gradOutput);
    }
    return;
  }

  THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);
  scalar_t norm = (reduction == Reduction::Mean ? 1./((scalar_t)THTensor_(nElement)(input)) : 1.);

  TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, input, scalar_t, target,
    scalar_t x = *input_data;
    scalar_t y = *target_data;
    *gradInput_data = - norm * (y - x) / ((1. - x + EPS) * (x + EPS)) * THTensor_(fastGetLegacy1dNoScalars)(gradOutput, 0);
  );

  if(weights)
    THTensor_(cmul)(gradInput, gradInput, weights);
}

#undef EPS

#endif
