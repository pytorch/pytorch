#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/SpatialSumPooling.c"
#else

void THNN_(SpatialSumPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode)
{
  THNN_(SpatialSumAveragePooling_updateOutput)
    (state, input, output, kW, kH, dW, dH, padW, padH, ceil_mode, false, false);
}

#endif
