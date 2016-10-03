#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/BCECriterion.c"
#else

#define EPS 1e-12

void THNN_(BCECriterion_updateOutput)(THNNState *state, THTensor *input,
				      THTensor *target, THTensor *output,
				      bool sizeAverage, THTensor *weights)
{
  THNN_CHECK_NELEMENT(input, target);
  THNN_CHECK_NELEMENT(input, weights);
  THNN_CHECK_DIM_SIZE(output, 1, 0, 1);
  real sum = 0;

  if(weights)
    TH_TENSOR_APPLY3(real, input, real, target, real, weights,
      real x = *input_data;
      real y = *target_data;
      real w = *weights_data;
      sum -= (log(x + EPS) * y + log(1. - x + EPS) * (1. - y)) * w;
    )
  else
    TH_TENSOR_APPLY2(real, input, real, target,
      real x = *input_data;
      real y = *target_data;
      sum -= log(x + EPS) * y + log(1. - x + EPS) * (1. - y);
    );


  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(BCECriterion_updateGradInput)(THNNState *state, THTensor *input,
					 THTensor *target, THTensor *gradInput,
					 bool sizeAverage, THTensor *weights)
{
  THNN_CHECK_NELEMENT(input, target);
  THNN_CHECK_NELEMENT(input, weights);

  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);

  THTensor_(resizeAs)(gradInput, input);

  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
    real x = *input_data;
    real y = *target_data;
    *gradInput_data = - norm * (y - x) / ((1. - x + EPS) * (x + EPS));
  );

  if(weights)
    THTensor_(cmul)(gradInput, gradInput, weights);
}

#undef EPS

#endif
