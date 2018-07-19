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
    int64_t kW, int64_t kH,
    int64_t dW, int64_t dH,
    int64_t padW, int64_t padH,
    int64_t adjW, int64_t adjH)
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
    int64_t kW, int64_t kH,
    int64_t dW, int64_t dH,
    int64_t padW, int64_t padH,
    int64_t adjW, int64_t adjH)
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
    int64_t kW, int64_t kH,
    int64_t dW, int64_t dH,
    int64_t padW, int64_t padH,
    int64_t adjW, int64_t adjH,
    accreal scale_)
{
THNN_(SpatialFullDilatedConvolution_accGradParameters)(
    state, input, gradOutput, gradWeight, gradBias, columns, ones,
    kW, kH, dW, dH, padW, padH, 1, 1, adjW, adjH, scale_);
}

#endif
