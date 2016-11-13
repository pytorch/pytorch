#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricMaxPooling.c"
#else

void THNN_(VolumetricMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          bool ceilMode)
{
  THNN_(VolumetricDilatedMaxPooling_updateOutput)(
          state, input, output, indices,
          kT, kW, kH, dT, dW, dH,
          pT, pW, pH, 1, 1, 1, ceilMode);
}

void THNN_(VolumetricMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  THNN_(VolumetricDilatedMaxPooling_updateGradInput)(
          state, input, gradOutput, gradInput, indices,
          dT, dW, dH, pT, pW, pH, 1, 1, 1);
}

#endif
