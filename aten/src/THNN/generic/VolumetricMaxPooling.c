#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricMaxPooling.c"
#else

void THNN_(VolumetricMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int64_t kT,
          int64_t kW,
          int64_t kH,
          int64_t dT,
          int64_t dW,
          int64_t dH,
          int64_t pT,
          int64_t pW,
          int64_t pH,
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
          int64_t kT,
          int64_t kW,
          int64_t kH,
          int64_t dT,
          int64_t dW,
          int64_t dH,
          int64_t pT,
          int64_t pW,
          int64_t pH,
          bool ceilMode)
{
  THNN_(VolumetricDilatedMaxPooling_updateGradInput)(
          state, input, gradOutput, gradInput, indices,
          kT, kW, kH, dT, dW, dH,
          pT, pW, pH, 1, 1, 1, ceilMode);
}

#endif
