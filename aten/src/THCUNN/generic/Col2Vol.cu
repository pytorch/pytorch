#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Col2Vol.cu"
#else

static inline void THNN_(Col2Vol_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         int outputDepth, int outputHeight, int outputWidth,
                         int kT, int kH, int kW, int dT, int dH, int dW,
                         int padT, int padH, int padW, int sT, int sH, int sW) {

  THArgCheck(kW > 0 && kH > 0 && kT > 0, 7,
             "kernel size should be greater than zero, but got kT: %d kH: %d kW: %d", kT, kH, kW);
  THArgCheck(sW > 0 && sH > 0 && sT > 0, 16,
             "stride should be greater than zero, but got sT: %d sH: %d sW: %d", sT, sH, sW);
  THArgCheck(dW > 0 && dH > 0 && dT > 0, 10,
             "dilation should be greater than zero, but got dT: %d dH: %d dW: %d", dT, dH, dW);

  int ndim = THCTensor_(nDimension)(state, input);
  THCUNN_argCheck(state, ndim == 2 || ndim == 3, 2, input,
                  "2D or 3D input tensor expected but got %s");

  int batch_dim = (ndim == 3) ? 0 : -1;
  long nInputPlane  = input->size[batch_dim + 1];
  long inputLength  = input->size[batch_dim + 2];

  long nOutputPlane = nInputPlane / (kW * kH * kT);

  if (outputWidth < 1 || outputHeight < 1 || outputDepth < 1) {
    THError("Given input size: (%lld x %lld). "
            "Calculated output size: (%lld x %d x %d x %d). Output size is too small",
            (long long)nInputPlane, (long long)inputLength, (long long)nOutputPlane, outputDepth, outputHeight, outputWidth);
  }
}

void THNN_(Col2Vol_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int outputDepth, int outputHeight, int outputWidth,
           int kT, int kH, int kW,
           int dT, int dH, int dW,
           int padT, int padH, int padW,
           int sT, int sH, int sW) {

  THCUNN_assertSameGPU(state, 2, input, output);

  THNN_(Col2Vol_shapeCheck)(state, input, NULL, outputDepth, outputHeight, outputWidth,
                            kT, kH, kW, dT, dH, dW, padT, padH, padW, sT, sH, sW);

  bool batched_input = true;
  if (input->nDimension == 2) {
      // Force batch
      batched_input = false;
      THCTensor_(resize3d)(state, input, 1, input->size[0], input->size[1]);
  }

  long batchSize = input->size[0];
  long nInputPlane = input->size[1];
  long nOutputPlane = nInputPlane / (kW * kH * kT);

  input = THCTensor_(newContiguous)(state, input);

  THCTensor_(resize5d)(state, output, batchSize, nOutputPlane, outputDepth, outputHeight, outputWidth);
  THCTensor_(zero)(state, output);

  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *output_n = THCTensor_(new)(state);

  int depth_col = (outputDepth + 2 * padT - (dT * (kT - 1) + 1)) / sT + 1;
  int height_col = (outputHeight + 2 * padH - (dH * (kH - 1) + 1)) / sH + 1;
  int width_col = (outputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;

  for (int elt = 0; elt < batchSize; elt++) {
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);

    col2vol<real, accreal>(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input_n),
      nOutputPlane,
      outputDepth, outputHeight, outputWidth,
      depth_col, height_col, width_col,
      kT, kH, kW,
      padT, padH, padW,
      sT, sH, sW,
      dT, dH, dW, THCTensor_(data)(state, output_n));
  }

  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);

  if (!batched_input) {
      THCTensor_(resize4d)(state, output, nOutputPlane, outputDepth, outputHeight, outputWidth);
  }
  THCTensor_(free)(state, input);
}

void THNN_(Col2Vol_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int kT, int kH, int kW,
           int dT, int dH, int dW,
           int padT, int padH, int padW,
           int sT, int sH, int sW) {

  THNN_(Vol2Col_updateOutput)(state, gradOutput, gradInput,
                              kT, kH, kW, dT, dH, dW, padT, padH, padW, sT, sH, sW);

}

#endif
