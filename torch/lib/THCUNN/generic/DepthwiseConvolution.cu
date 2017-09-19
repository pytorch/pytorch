#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/DepthwiseConvolution.cu"
#else

void THNN_(DepthwiseConvolution_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,
                  THCTensor *columns,
                  THCTensor *ones,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH) {
  // TODO: implement
}

void THNN_(DepthwiseConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  THCTensor *gradColumns,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH) {
  // TODO: implement
}

void THNN_(DepthwiseConvolution_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  THCTensor *columns,
                  THCTensor *ones,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH,
                  accreal scale) {

  // TODO: implement
}

#endif
