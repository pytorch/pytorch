#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricMaxPooling.cu"
#else

void THNN_(VolumetricMaxPooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCIndexTensor *indices,
           int64_t kT, int64_t kW, int64_t kH,
           int64_t dT, int64_t dW, int64_t dH,
           int64_t padT, int64_t padW, int64_t padH,
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
           int64_t kT, int64_t kW, int64_t kH,
           int64_t dT, int64_t dW, int64_t dH,
           int64_t padT, int64_t padW, int64_t padH,
           bool ceilMode)
{
  THNN_(VolumetricDilatedMaxPooling_updateGradInput)(
        state, input, gradOutput, gradInput, indices,
        kT, kW, kH, dT, dW, dH, padT, padW, padH,
        1, 1, 1, ceilMode);

}

#endif
