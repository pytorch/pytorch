#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/HardShrink.c"
#else

void THNN_(HardShrink_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real lambda)
{
  THTensor_(resizeAs)(output, input);

  TH_TENSOR_APPLY2(real, output, real, input,
    if (*input_data > lambda)
      *output_data = *input_data;
    else if (*input_data < -lambda)
      *output_data = *input_data;
    else
      *output_data = 0;
  );
}

void THNN_(HardShrink_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real lambda)
{
  THNN_CHECK_NELEMENT(input, gradOutput);
  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
    if (*input_data > lambda || *input_data < -lambda)
      *gradInput_data = *gradOutput_data;
    else
      *gradInput_data = 0;
  );
}

#endif
