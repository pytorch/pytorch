#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ELU.c"
#else

void THNN_(ELU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal alpha_,
          accreal scale,
          accreal input_scale,
          bool inplace)
{
  scalar_t negcoef = TH_CONVERT_ACCREAL_TO_REAL(alpha_ * scale);
  scalar_t poscoef = TH_CONVERT_ACCREAL_TO_REAL(scale * input_scale);
  scalar_t negiptcoef = TH_CONVERT_ACCREAL_TO_REAL(input_scale);
  if (inplace) {
    TH_TENSOR_APPLY(scalar_t, input,
      *input_data = *input_data <= 0 ? (exp(*input_data * negiptcoef)-1) * negcoef : *input_data * poscoef;
    );
    THTensor_(set)(output, input);
  } else {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(scalar_t, input, scalar_t, output,
      *output_data = *input_data <= 0 ? (exp(*input_data * negiptcoef)-1) * negcoef : *input_data * poscoef;
    );
  }
}

void THNN_(ELU_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          accreal alpha_,
          accreal scale,
          accreal input_scale)
{
  scalar_t negcoef = TH_CONVERT_ACCREAL_TO_REAL(alpha_ * scale);
  scalar_t poscoef = TH_CONVERT_ACCREAL_TO_REAL(scale * input_scale);
  scalar_t negiptcoef = TH_CONVERT_ACCREAL_TO_REAL(input_scale);
  THNN_CHECK_NELEMENT(output, gradOutput);
  THTensor_(resizeAs)(gradInput, output);
  TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, gradOutput, scalar_t, output,
    *gradInput_data = *output_data <= 0 ? *gradOutput_data * negiptcoef * (*output_data + negcoef) : *gradOutput_data * poscoef;
  );
}

#endif
