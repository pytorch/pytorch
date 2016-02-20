#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ELU.c"
#else

void THNN_(ELU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real alpha)
{
  THTensor_(resizeAs)(output, input);
  TH_TENSOR_APPLY2(real, input, real, output,
    *output_data = *input_data <= 0 ? (exp(*input_data)-1)*alpha : *input_data;
  );
}

void THNN_(ELU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          real alpha)
{
  THTensor_(resizeAs)(gradInput, output);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
    *gradInput_data = *output_data <= 0 ? *gradOutput_data * (*output_data + alpha) : *gradOutput_data;
  );
}

#endif
