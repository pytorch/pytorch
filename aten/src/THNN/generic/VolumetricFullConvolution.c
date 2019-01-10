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
  int kT, int kW, int kH,   // kenerl size
  int dT, int dW, int dH,   // stride of the convolution
  int pT, int pW, int pH,   // padding
  int aT, int aW, int aH)   // extra output adjustment
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
  int kT, int kW, int kH,   // kenerl size
  int dT, int dW, int dH,   // stride
  int pT, int pW, int pH,   // padding
  int aT, int aW, int aH)   // extra output adjustment
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
  int kT, int kW, int kH,   // kenerl size
  int dT, int dW, int dH,   // stride
  int pT, int pW, int pH,   // padding
  int aT, int aW, int aH,   // extra output adjustment
  accreal scale_)
{
  THNN_(VolumetricFullDilatedConvolution_accGradParameters)(
      state, input, gradOutput, gradWeight, gradBias, finput, fgradInput,
      kT, kW, kH, dT, dW, dH, pT, pW, pH, 1, 1, 1, aT, aW, aH, scale_);
}

#endif
