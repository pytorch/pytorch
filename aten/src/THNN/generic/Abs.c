#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Abs.c"
#else

void THNN_(Abs_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  THTensor_(resizeAs)(output, input);
  THTensor_(abs)(output, input);
}

void THNN_(Abs_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput)
{
  THNN_CHECK_NELEMENT(input, gradOutput);
  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
    real z = *input_data;
    *gradInput_data = *gradOutput_data * (z >= 0 ? 1 : -1);
  );
}

#endif
