#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LogSigmoid.c"
#else

void THNN_(LogSigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *buffer)
{
  THTensor_(resizeAs)(output, input);
  THTensor_(resizeAs)(buffer, input);

  TH_TENSOR_APPLY3(real, output, real, input, real, buffer,
    real z = exp(-*input_data);
    *buffer_data = z;
    *output_data = -log(1. + z);
  );
}

void THNN_(LogSigmoid_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *buffer)
{
  THNN_CHECK_NELEMENT(input, gradOutput);
  THTensor_(resizeAs)(gradInput, buffer);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, buffer,
    real z = *buffer_data;
    *gradInput_data = *gradOutput_data * z / (1. + z);
  );
}

#endif
