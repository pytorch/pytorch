#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ReLU6.c"
#else

void THNN_(ReLU6_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          bool inplace)
{
  if (inplace)
  {
    TH_TENSOR_APPLY(real, input,
      if (*input_data <= 0)
        *input_data = 0;
      else if (*input_data >= 6)
        *input_data = 6;
    );
    THTensor_(set)(output, input);
  }
  else
  {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(real, output, real, input,
      *output_data =
         (*input_data > 0) ? ((*input_data < 6) ? *input_data : 6) : 0;
    );
  }
}

void THNN_(ReLU6_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          bool inplace)
{
  if (inplace)
  {
    TH_TENSOR_APPLY2(real, gradOutput, real, input,
      if ((*input_data) <= 0 || (*input_data) >= 6)
        *gradOutput_data = 0;
    );
    THTensor_(set)(gradInput, gradOutput);
  }
  else
  {
    THTensor_(resizeAs)(gradInput, input);
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
      if ((*input_data) > 0 && (*input_data) < 6)
        *gradInput_data = *gradOutput_data;
      else
        *gradInput_data = 0;
    );
  }
}

#endif
