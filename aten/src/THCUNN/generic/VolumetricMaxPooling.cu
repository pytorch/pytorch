#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricMaxPooling.cu"
#else

void THNN_(VolumetricMaxPooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCIndexTensor *indices,
           int kT, int kW, int kH,
           int dT, int dW, int dH,
           int padT, int padW, int padH,
           bool ceilMode)
{
  THNN_(VolumetricDilatedMaxPooling_updateOutput)(
        state, input, output, indices,
        kT, kW, kH, dT, dW, dH, padT, padW, padH,
        1, 1, 1, ceilMode);

}

void THNN_(VolumetricMaxPooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCIndexTensor *indices,
           int kT, int kW, int kH,
           int dT, int dW, int dH,
           int padT, int padW, int padH,
           bool ceilMode)
{
  THNN_(VolumetricDilatedMaxPooling_updateGradInput)(
        state, input, gradOutput, gradInput, indices,
        kT, kW, kH, dT, dW, dH, padT, padW, padH,
        1, 1, 1, ceilMode);

}

#endif
