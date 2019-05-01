#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/SoftShrink.c"
#else

void THNN_(SoftShrink_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal lambda_)
{
  scalar_t lambda = TH_CONVERT_ACCREAL_TO_REAL(lambda_);
  THTensor_(resizeAs)(output, input);

  TH_TENSOR_APPLY2(scalar_t, output, scalar_t, input,
    if ((*input_data) > lambda)
     *output_data = *input_data - lambda;
    else if ((*input_data) < -lambda)
     *output_data = *input_data + lambda;
    else
     *output_data = 0;
  );
}

void THNN_(SoftShrink_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accreal lambda_)
{
  scalar_t lambda = TH_CONVERT_ACCREAL_TO_REAL(lambda_);
  THNN_CHECK_NELEMENT(input, gradOutput);
  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, gradOutput, scalar_t, input,
    if ((*input_data) > lambda || (*input_data) < -lambda)
      *gradInput_data = (*gradOutput_data);
    else
      *gradInput_data = 0;
  );
}

#endif
