#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMaxPooling.c"
#else

void THNN_(SpatialMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode)
{
  THNN_(SpatialDilatedMaxPooling_updateOutput)(
      state, input, output, indices,
      kW, kH, dW, dH, padW, padH, 1, 1, ceil_mode
    );
}

void THNN_(SpatialMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode)
{
  THNN_(SpatialDilatedMaxPooling_updateGradInput)(
      state, input, gradOutput, gradInput, indices,
      kW, kH, dW, dH, padW, padH, 1, 1, ceil_mode
    );
}

#endif
