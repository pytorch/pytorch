#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Vol2Col.cu"
#else

static inline void THNN_(Vol2Col_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         int kT, int kH, int kW, int dT, int dH, int dW,
                         int padT, int padH, int padW, int sT, int sH, int sW) {

  THArgCheck(kW > 0 && kH > 0 && kT > 0, 4,
             "kernel size should be greater than zero, but got kT: %d kH: %d kW: %d", kT, kH, kW);
  THArgCheck(dW > 0 && dH > 0 && kT > 0, 7,
             "dilation should be greater than zero, but got dT: %d dH: %d dW: %d", dT, dH, dW);
  THArgCheck(padW >= 0 && padH >= 0 && padT >=0, 10,
             "padding should be non-negative, but got padT: %d padH: %d padW: %d", padT, padH, padW);
  THArgCheck(sW > 0 && sH > 0 && sT > 0, 13,
             "stride should be greater than zero, but got sT: %d sH: %d sW: %d", sT, sH, sW);

  int ndim = THCTensor_(nDimension)(state, input);
  THCUNN_argCheck(state, ndim == 4 || ndim == 5, 2, input,
                  "4D or 5D input tensor expected but got: %s");

  int dim_batch = 0;
  if (ndim == 4) {
    dim_batch = -1;
  }
  int nInputPlane  = THCTensor_(size)(state, input, dim_batch + 1);
  int inputDepth   = THCTensor_(size)(state, input, dim_batch + 2);
  int inputHeight  = THCTensor_(size)(state, input, dim_batch + 3);
  int inputWidth   = THCTensor_(size)(state, input, dim_batch + 4);
  int outputDepth  = (inputDepth + 2 * padT - (dT * (kT - 1) + 1)) / sT + 1;
  int outputHeight = (inputHeight + 2 * padH - (dH * (kH - 1) + 1)) / sH + 1;
  int outputWidth  = (inputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;
  int nOutputPlane = nInputPlane * kW * kH * kT;
  int outputLength = outputHeight * outputWidth;

  if (outputWidth < 1 || outputHeight < 1 || outputDepth < 1) {
    THError("Given input size: (%d x %d x %d x %d). "
            "Calculated output size: (%d x %d). Output size is too small",
            nInputPlane, inputDepth, inputHeight, inputWidth, nOutputPlane, outputLength);
  }
}

void THNN_(Vol2Col_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int kT, int kH, int kW,
           int dT, int dH, int dW,
           int padT, int padH, int padW,
           int sT, int sH, int sW) {

  THCUNN_assertSameGPU(state, 2, input, output);

  THNN_(Vol2Col_shapeCheck)(state, input, NULL, kT, kH, kW, dT, dH, dW, padT, padH, padW, sT, sH, sW);

  input = THCTensor_(newContiguous)(state, input);
  bool batched_input = true;
  if (input->nDimension == 4) {
    batched_input = false;
    THCTensor_(resize5d)(state, input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
  }

  int batchSize    = THCTensor_(size)(state, input, 0);
  int nInputPlane  = THCTensor_(size)(state, input, 1);
  int inputDepth   = THCTensor_(size)(state, input, 2);
  int inputHeight  = THCTensor_(size)(state, input, 3);
  int inputWidth   = THCTensor_(size)(state, input, 4);

  int outputDepth  = (inputDepth + 2 * padT - (dT * (kT - 1) + 1)) / sT + 1;
  int outputHeight = (inputHeight + 2 * padH - (dH * (kH - 1) + 1)) / sH + 1;
  int outputWidth  = (inputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;
  int nOutputPlane = nInputPlane * kW * kH * kT;
  int outputLength = outputDepth * outputHeight * outputWidth;

  THCTensor_(resize3d)(state, output, batchSize, nOutputPlane, outputLength);
  THCTensor_(zero)(state, output);

  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *output_n = THCTensor_(new)(state);

  for (int elt = 0; elt < batchSize; elt++) {
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);

    vol2col(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input_n),
      nInputPlane, inputDepth, inputHeight, inputWidth, kT, kH, kW, padT, padH, padW, sT, sH, sW,
      dT, dH, dW, THCTensor_(data)(state, output_n));
  }

  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);

  if (!batched_input) {
    THCTensor_(resize2d)(state, output, nOutputPlane, outputLength);
  }
  THCTensor_(free)(state, input);
}

void THNN_(Vol2Col_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int inputDepth, int inputHeight, int inputWidth,
           int kT, int kH, int kW,
           int dT, int dH, int dW,
           int padT, int padH, int padW,
           int sT, int sH, int sW) {

  THNN_(Col2Vol_updateOutput)(state, gradOutput, gradInput,
                              inputDepth, inputHeight, inputWidth,
                              kT, kH, kW, dT, dH, dW,
                              padT, padH, padW, sT, sH, sW);
}

#endif
