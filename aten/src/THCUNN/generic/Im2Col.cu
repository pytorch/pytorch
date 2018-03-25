#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Im2Col.cu"
#else

static inline void THNN_(Im2Col_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         int kH, int kW, int dH, int dW,
                         int padH, int padW, int sH, int sW) {

  THArgCheck(kW > 0 && kH > 0, 4,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 6,
             "dilation should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THArgCheck(padW >= 0 && padH >= 0, 8,
             "padding should be non-negative, but got padH: %d padW: %d", padH, padW);
  THArgCheck(sW > 0 && sH > 0, 10,
             "stride should be greater than zero, but got sH: %d sW: %d", sH, sW);

  int ndim = THCTensor_(nDimension)(state, input);
  THCUNN_argCheck(state, ndim == 3 || ndim == 4, 2, input,
                  "3D or 4D input tensor expected but got: %s");

  int dim_batch = 0;
  if (ndim == 3) {
    dim_batch = -1;
  }
  int nInputPlane  = THCTensor_(size)(state, input, dim_batch + 1);
  int inputHeight  = THCTensor_(size)(state, input, dim_batch + 2);
  int inputWidth   = THCTensor_(size)(state, input, dim_batch + 3);
  int outputHeight = (inputHeight + 2 * padH - (dH * (kH - 1) + 1)) / sH + 1;
  int outputWidth  = (inputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;
  int nOutputPlane = nInputPlane * kW * kH;
  int outputLength = outputHeight * outputWidth;

  if (outputWidth < 1 || outputHeight < 1) {
    THError("Given input size: (%d x %d x %d). "
            "Calculated output size: (%d x %d). Output size is too small",
            nInputPlane, inputHeight, inputWidth, nOutputPlane, outputLength);
  }
}

void THNN_(Im2Col_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int kH, int kW,
           int dH, int dW,
           int padH, int padW,
           int sH, int sW) {

  THCUNN_assertSameGPU(state, 2, input, output);

  THNN_(Im2Col_shapeCheck)(state, input, NULL, kH, kW, dH, dW, padH, padW, sH, sW);

  input = THCTensor_(newContiguous)(state, input);
  bool batched_input = true;
  if (input->nDimension == 3) {
    batched_input = false;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
  }

  int batchSize    = THCTensor_(size)(state, input, 0);
  int nInputPlane  = THCTensor_(size)(state, input, 1);
  int inputHeight  = THCTensor_(size)(state, input, 2);
  int inputWidth   = THCTensor_(size)(state, input, 3);

  int outputHeight = (inputHeight + 2 * padH - (dH * (kH - 1) + 1)) / sH + 1;
  int outputWidth  = (inputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;
  int nOutputPlane = nInputPlane * kW * kH;
  int outputLength = outputHeight * outputWidth;

  THCTensor_(resize3d)(state, output, batchSize, nOutputPlane, outputLength);
  THCTensor_(zero)(state, output);

  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *output_n = THCTensor_(new)(state);

  for (int elt = 0; elt < batchSize; elt++) {
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);

    im2col(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input_n),
      nInputPlane, inputHeight, inputWidth,
      outputHeight, outputWidth,
      kH, kW, padH, padW, sH, sW,
      dH, dW, THCTensor_(data)(state, output_n));
  }

  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);

  if (!batched_input) {
    THCTensor_(resize2d)(state, output, nOutputPlane, outputLength);
  }
  THCTensor_(free)(state, input);
}

void THNN_(Im2Col_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int inputHeight, int inputWidth,
           int kH, int kW,
           int dH, int dW,
           int padH, int padW,
           int sH, int sW) {

  THNN_(Col2Im_updateOutput)(state, gradOutput, gradInput,
                             inputHeight, inputWidth,
                             kH, kW, dH, dW,
                             padH, padW, sH, sW);
}

#endif
