#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialFullConvolution.c"
#else

void THNN_(SpatialFullConvolution_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    THTensor *weight,
    THTensor *bias,
    THTensor *columns,
    THTensor *ones,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int adjW, int adjH)
{
  THNN_(SpatialFullDilatedConvolution_updateOutput)(
    state, input, output, weight, bias, columns, ones,
    kW, kH, dW, dH, padW, padH, 1, 1, adjW, adjH);
  }

void THNN_(SpatialFullConvolution_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    THTensor *weight,
    THTensor *gradColumns,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int adjW, int adjH)
{
  THNN_(SpatialFullDilatedConvolution_updateGradInput)(
    state, input, gradOutput, gradInput, weight, gradColumns,
    kW, kH, dW, dH, padW, padH, 1, 1, adjW, adjH);
}

void THNN_(SpatialFullConvolution_accGradParameters)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradWeight,
    THTensor *gradBias,
    THTensor *columns,
    THTensor *ones,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int adjW, int adjH,
    accreal scale_)
{
THNN_(SpatialFullDilatedConvolution_accGradParameters)(
    state, input, gradOutput, gradWeight, gradBias, columns, ones,
    kW, kH, dW, dH, padW, padH, 1, 1, adjW, adjH, scale_);
}

#endif
