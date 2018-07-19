#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMaxPooling.c"
#else

void THNN_(SpatialMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int64_t kW,
          int64_t kH,
          int64_t dW,
          int64_t dH,
          int64_t padW,
          int64_t padH,
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
          int64_t kW,
          int64_t kH,
          int64_t dW,
          int64_t dH,
          int64_t padW,
          int64_t padH,
          bool ceil_mode)
{
  THNN_(SpatialDilatedMaxPooling_updateGradInput)(
      state, input, gradOutput, gradInput, indices,
      kW, kH, dW, dH, padW, padH, 1, 1, ceil_mode
    );
}

#endif
