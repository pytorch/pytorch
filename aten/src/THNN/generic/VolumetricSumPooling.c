#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/VolumetricSumPooling.c"
#else

void THNN_(VolumetricSumPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int padT,
          int padW,
          int padH,
          bool ceil_mode)
{
  THNN_(VolumetricSumAveragePooling_updateOutput)
    (state, input, output, kT, kW, kH, dT, dW, dH, padT, padW, padH, ceil_mode, false, false);
}

 #endif
