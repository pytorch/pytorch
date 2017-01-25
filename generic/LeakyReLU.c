#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LeakyReLU.c"
#else

void THNN_(LeakyReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real negval,
          bool inplace)
{
  if (inplace)
  {
    TH_TENSOR_APPLY(real, input,
      if (*input_data <= 0)
        *input_data *= negval;
    );
    THTensor_(set)(output, input);
  }
  else
  {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(real, output, real, input,
      *output_data = *input_data > 0 ? *input_data : *input_data * negval;
    );
  }
}

void THNN_(LeakyReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real negval,
          bool inplace)
{
  THNN_CHECK_NELEMENT(input, gradOutput);
  if (inplace)
  {
    TH_TENSOR_APPLY2(real, gradOutput, real, input,
      if (*input_data <= 0)
        *gradOutput_data *= negval;
    );
    THTensor_(set)(gradInput, gradOutput);
  }
  else
  {
    THTensor_(resizeAs)(gradInput, input);
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
      *gradInput_data = *input_data > 0 ? *gradOutput_data : *gradOutput_data * negval;
    );
  }
}

#endif
