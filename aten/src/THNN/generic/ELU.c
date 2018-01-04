#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ELU.c"
#else

void THNN_(ELU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal alpha_,
          accreal scale,
          bool inplace)
{
  real negcoef = TH_CONVERT_ACCREAL_TO_REAL(alpha_ * scale);
  real poscoef = TH_CONVERT_ACCREAL_TO_REAL(scale);
  if (inplace) {
    TH_TENSOR_APPLY(real, input,
      *input_data = *input_data <= 0 ? (exp(*input_data)-1) * negcoef : *input_data * poscoef;
    );
    THTensor_(set)(output, input);
  } else {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(real, input, real, output,
      *output_data = *input_data <= 0 ? (exp(*input_data)-1) * negcoef : *input_data * poscoef;
    );
  }
}

void THNN_(ELU_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          accreal alpha_,
          accreal scale)
{
  real negcoef = TH_CONVERT_ACCREAL_TO_REAL(alpha_ * scale);
  real poscoef = TH_CONVERT_ACCREAL_TO_REAL(scale);
  THNN_CHECK_NELEMENT(output, gradOutput);
  THTensor_(resizeAs)(gradInput, output);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
    *gradInput_data = *output_data <= 0 ? *gradOutput_data * (*output_data + negcoef) : *gradOutput_data * poscoef;
  );
}

#endif
