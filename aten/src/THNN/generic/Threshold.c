#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Threshold.c"
#else

void THNN_(Threshold_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal threshold_,
          accreal val_,
          bool inplace)
{
  real threshold = TH_CONVERT_ACCREAL_TO_REAL(threshold_);
  real val = TH_CONVERT_ACCREAL_TO_REAL(val_);
  if (inplace)
  {
    TH_TENSOR_APPLY(real, input,
      if (*input_data <= threshold)
        *input_data = val;
    );
    THTensor_(set)(output, input);
  }
  else
  {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(real, output, real, input,
      *output_data = (*input_data > threshold) ? *input_data : val;
    );
  }
}

void THNN_(Threshold_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accreal threshold_,
          accreal val_,
          bool inplace)
{
  real threshold = TH_CONVERT_ACCREAL_TO_REAL(threshold_);
  THNN_CHECK_NELEMENT(input, gradOutput);
  if (inplace)
  {
    TH_TENSOR_APPLY2(real, gradOutput, real, input,
      if ((*input_data) <= threshold)
        *gradOutput_data = 0;
    );
    THTensor_(set)(gradInput, gradOutput);
  }
  else
  {
    THTensor_(resizeAs)(gradInput, input);
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
      if ((*input_data) > threshold)
        *gradInput_data = *gradOutput_data;
      else
        *gradInput_data = 0;
    );
  }
}

#endif
