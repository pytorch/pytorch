#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricFullConvolution.cu"
#else

void THNN_(VolumetricFullConvolution_updateOutput)(
       THCState *state,
       THCTensor  *input,
       THCTensor  *output,
       THCTensor  *weight,
       THCTensor  *bias,
       THCTensor  *finput,
       THCTensor  *fgradInput,
       int64_t kT, int64_t kW, int64_t kH,
       int64_t dT, int64_t dW, int64_t dH,
       int64_t padT, int64_t padW, int64_t padH,
       int64_t adjT, int64_t adjW, int64_t adjH)
{
  THNN_(VolumetricFullDilatedConvolution_updateOutput)(
       state, input, output, weight, bias, finput, fgradInput,
       kT, kW, kH, dT, dW, dH, padT, padW, padH, 1, 1, 1, adjT, adjW, adjH);
}

void THNN_(VolumetricFullConvolution_updateGradInput)(
       THCState *state,
       THCTensor  *input,
       THCTensor  *gradOutput,
       THCTensor  *gradInput,
       THCTensor  *weight,
       THCTensor  *finput,
       THCTensor  *fgradInput,
       int64_t kT, int64_t kW, int64_t kH,
       int64_t dT, int64_t dW, int64_t dH,
       int64_t padT, int64_t padW, int64_t padH,
       int64_t adjT, int64_t adjW, int64_t adjH)
{
  THNN_(VolumetricFullDilatedConvolution_updateGradInput)(
       state, input, gradOutput, gradInput, weight, finput, fgradInput,
       kT, kW, kH, dT, dW, dH, padT, padW, padH, 1, 1, 1, adjT, adjW, adjH);
}


void THNN_(VolumetricFullConvolution_accGradParameters)(
           THCState *state,
           THCTensor  *input,
           THCTensor  *gradOutput,
           THCTensor  *gradWeight,
           THCTensor  *gradBias,
           THCTensor  *finput,
           THCTensor  *fgradInput,
           int64_t kT, int64_t kW, int64_t kH,
           int64_t dT, int64_t dW, int64_t dH,
           int64_t padT, int64_t padW, int64_t padH,
           int64_t adjT, int64_t adjW, int64_t adjH,
           accreal scale_)
{
  THNN_(VolumetricFullDilatedConvolution_accGradParameters)(
       state, input, gradOutput, gradWeight, gradBias, finput, fgradInput,
       kT, kW, kH, dT, dW, dH, padT, padW, padH, 1, 1, 1, adjT, adjW, adjH, scale_);
}

#endif
