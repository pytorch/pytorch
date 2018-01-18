#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Col2Im.cu"
#else

static inline void THNN_(Col2Im_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         int outputHeight, int outputWidth,
                         int kH, int kW, int dH, int dW,
                         int padH, int padW, int sH, int sW) {

  THArgCheck(kW > 0 && kH > 0, 6,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(sW > 0 && sH > 0, 12,
             "stride should be greater than zero, but got sH: %d sW: %d", sH, sW);
  THArgCheck(dW > 0 && dH > 0, 8,
             "dilation should be greater than zero, but got dH: %d dW: %d", dH, dW);

  int ndim = THCTensor_(nDimension)(state, input);
  THCUNN_argCheck(state, ndim == 2 || ndim == 3, 2, input,
                  "2D or 3D input tensor expected but got %s");

  int batch_dim = (ndim == 4) ? 0 : -1;
  long nInputPlane  = input->size[batch_dim + 1];
  long inputLength  = input->size[batch_dim + 2];

  long nOutputPlane = nInputPlane / (kW * kH);

  if (outputWidth < 1 || outputHeight < 1) {
    THError("Given input size: (%lld x %lld). "
            "Calculated output size: (%lld x %d x %d). Output size is too small",
            (long long)nInputPlane, (long long)inputLength, (long long)nOutputPlane, outputHeight, outputWidth);
  }
}

void THNN_(Col2Im_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int outputHeight, int outputWidth,
           int kH, int kW,
           int dH, int dW,
           int padH, int padW,
           int sH, int sW) {

  THCUNN_assertSameGPU(state, 2, input, output);

  THNN_(Col2Im_shapeCheck)(state, input, NULL, outputHeight, outputWidth,
                           kH, kW, dH, dW, padH, padW, sH, sW);

  bool batched_input = true;
  if (input->nDimension == 2) {
      // Force batch
      batched_input = false;
      THCTensor_(resize3d)(state, input, 1, input->size[0], input->size[1]);
  }

  long batchSize = input->size[0];
  long nInputPlane = input->size[1];
  long nOutputPlane = nInputPlane / (kW * kH);

  input = THCTensor_(newContiguous)(state, input);

  THCTensor_(resize4d)(state, output, batchSize, nOutputPlane, outputHeight, outputWidth);
  THCTensor_(zero)(state, output);

  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *output_n = THCTensor_(new)(state);

  int height_col = (outputHeight + 2 * padH - (dH * (kH - 1) + 1)) / sH + 1;
  int width_col = (outputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;

  for (int elt = 0; elt < batchSize; elt++) {
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);

    col2im<real, accreal>(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input_n),
      nOutputPlane,
      outputHeight, outputWidth,
      height_col, width_col,
      kH, kW,
      padH, padW,
      sH, sW,
      dH, dW, THCTensor_(data)(state, output_n));
  }

  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);

  if (!batched_input) {
      THCTensor_(resize3d)(state, output, nOutputPlane, outputHeight, outputWidth);
  }
  THCTensor_(free)(state, input);
}

void THNN_(Col2Im_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int kH, int kW,
           int dH, int dW,
           int padH, int padW,
           int sH, int sW) {

  THNN_(Im2Col_updateOutput)(state, gradOutput, gradInput,
                             kH, kW, dH, dW, padH, padW, sH, sW);

}

#endif
