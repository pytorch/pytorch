#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/LeakyReLU.c"
#else

void THNN_(LeakyReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal negval_,
          bool inplace)
{
  scalar_t negval = TH_CONVERT_ACCREAL_TO_REAL(negval_);
  if (inplace)
  {
    TH_TENSOR_APPLY(scalar_t, input,
      if (*input_data <= 0)
        *input_data *= negval;
    );
    THTensor_(set)(output, input);
  }
  else
  {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(scalar_t, output, scalar_t, input,
      const scalar_t r = (*input_data > 0) ? 1 : negval;
      *output_data = *input_data * r;
    );
  }
}

void THNN_(LeakyReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accreal negval_,
          bool inplace)
{
  scalar_t negval = TH_CONVERT_ACCREAL_TO_REAL(negval_);
  THNN_CHECK_NELEMENT(input, gradOutput);
  if (inplace)
  {
    TH_TENSOR_APPLY2(scalar_t, gradOutput, scalar_t, input,
      if (*input_data <= 0)
        *gradOutput_data *= negval;
    );
    THTensor_(set)(gradInput, gradOutput);
  }
  else
  {
    THTensor_(resizeAs)(gradInput, input);
    TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, gradOutput, scalar_t, input,
      *gradInput_data = *input_data > 0 ? *gradOutput_data : *gradOutput_data * negval;
    );
  }
}

#endif
