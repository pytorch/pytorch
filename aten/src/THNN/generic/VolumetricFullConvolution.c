#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricFullConvolution.c"
#else

void THNN_(VolumetricFullConvolution_updateOutput)(
  THNNState *state,
  THTensor *input,          // 4D or 5D (batch) tensor
  THTensor *output,
  THTensor *weight,         // weight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
  THTensor *bias,
  THTensor *finput,         // internal columns buffer
  THTensor *fgradInput,     // internal ones buffer
  int64_t kT, int64_t kW, int64_t kH,   // kenerl size
  int64_t dT, int64_t dW, int64_t dH,   // stride of the convolution
  int64_t pT, int64_t pW, int64_t pH,   // padding
  int64_t aT, int64_t aW, int64_t aH)   // extra output adjustment
{
  THNN_(VolumetricFullDilatedConvolution_updateOutput)(
      state, input, output, weight, bias, finput, fgradInput,
      kT, kW, kH, dT, dW, dH, pT, pW, pH, 1, 1, 1, aT, aW, aH);
}

void THNN_(VolumetricFullConvolution_updateGradInput)(
  THNNState *state,
  THTensor *input,
  THTensor *gradOutput,
  THTensor *gradInput,
  THTensor *weight,
  THTensor *finput,
  THTensor *fgradInput,     // only used by cuda impl
  int64_t kT, int64_t kW, int64_t kH,   // kenerl size
  int64_t dT, int64_t dW, int64_t dH,   // stride
  int64_t pT, int64_t pW, int64_t pH,   // padding
  int64_t aT, int64_t aW, int64_t aH)   // extra output adjustment
{
  THNN_(VolumetricFullDilatedConvolution_updateGradInput)(
      state, input, gradOutput, gradInput, weight, finput, fgradInput,
      kT, kW, kH, dT, dW, dH, pT, pW, pH, 1, 1, 1, aT, aW, aH);
}

void THNN_(VolumetricFullConvolution_accGradParameters)(
  THNNState *state,
  THTensor *input,
  THTensor *gradOutput,
  THTensor *gradWeight,
  THTensor *gradBias,
  THTensor *finput,
  THTensor *fgradInput,
  int64_t kT, int64_t kW, int64_t kH,   // kenerl size
  int64_t dT, int64_t dW, int64_t dH,   // stride
  int64_t pT, int64_t pW, int64_t pH,   // padding
  int64_t aT, int64_t aW, int64_t aH,   // extra output adjustment
  accreal scale_)
{
  THNN_(VolumetricFullDilatedConvolution_accGradParameters)(
      state, input, gradOutput, gradWeight, gradBias, finput, fgradInput,
      kT, kW, kH, dT, dW, dH, pT, pW, pH, 1, 1, 1, aT, aW, aH, scale_);
}

#endif
