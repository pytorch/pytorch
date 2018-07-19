#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialFullConvolution.cu"
#else

void THNN_(SpatialFullConvolution_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias,
           THCTensor *columns,
           THCTensor *ones,
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
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *weight,
           THCTensor *gradColumns,
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
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *columns,
           THCTensor *ones,
           int64_t kW, int64_t kH,
           int64_t dW, int64_t dH,
           int64_t padW, int64_t padH,
           int64_t adjW, int64_t adjH,
           accreal scale_)
{
  THNN_(SpatialFullDilatedConvolution_accGradParameters)(
      state, input, gradOutput, gradWeight, gradBias,
      columns, ones,
      kW, kH, dW, dH, padW, padH, 1, 1, adjW, adjH, scale_);
}

#endif