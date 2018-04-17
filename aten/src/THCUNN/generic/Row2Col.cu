#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Row2Col.cu"
#else

static inline void THNN_(Row2Col_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         int kW, int dW,
                         int padW, int sW) {

  THArgCheck(kW > 0, 4,
             "kernel size should be greater than zero, but got kW: %d", kW);
  THArgCheck(dW > 0, 5,
             "dilation should be greater than zero, but got dW: %d", dW);
  THArgCheck(padW >= 0, 6,
             "padding should be non-negative, but got padW: %d", padW);
  THArgCheck(sW > 0, 7,
             "stride should be greater than zero, but got sW: %d", sW);

  int ndim = THCTensor_(nDimension)(state, input);
  THCUNN_argCheck(state, ndim == 2 || ndim == 3, 2, input,
                  "2D or 3D input tensor expected but got: %s");

  int dim_batch = 0;
  if (ndim == 2) {
    dim_batch = -1;
  }
  int nInputPlane  = THCTensor_(size)(state, input, dim_batch + 1);
  int inputWidth   = THCTensor_(size)(state, input, dim_batch + 2);
  int outputWidth  = (inputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;
  int nOutputPlane = nInputPlane * kW;
  int outputLength = outputWidth;

  if (outputWidth < 1) {
    THError("Given input size: (%d x %d). "
            "Calculated output size: (%d x %d). Output size is too small",
            nInputPlane, inputWidth, nOutputPlane, outputLength);
  }
}

void THNN_(Row2Col_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int kW,
           int dW,
           int padW,
           int sW) {

  THCUNN_assertSameGPU(state, 2, input, output);

  THNN_(Row2Col_shapeCheck)(state, input, NULL, kW, dW, padW, sW);

  input = THCTensor_(newContiguous)(state, input);
  bool batched_input = true;
  if (input->nDimension == 2) {
    batched_input = false;
    THCTensor_(resize3d)(state, input, 1, input->size[0], input->size[1]);
  }

  int batchSize    = THCTensor_(size)(state, input, 0);
  int nInputPlane  = THCTensor_(size)(state, input, 1);
  int inputWidth   = THCTensor_(size)(state, input, 2);

  int outputWidth  = (inputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;
  int nOutputPlane = nInputPlane * kW;
  int outputLength = outputWidth;

  THCTensor_(resize3d)(state, output, batchSize, nOutputPlane, outputLength);
  THCTensor_(zero)(state, output);

  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *output_n = THCTensor_(new)(state);

  for (int elt = 0; elt < batchSize; elt++) {
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);

    row2col(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input_n),
      nInputPlane, inputWidth, kW, padW, sW,
      dW, THCTensor_(data)(state, output_n));
  }

  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);

  if (!batched_input) {
    THCTensor_(resize2d)(state, output, nOutputPlane, outputLength);
  }
  THCTensor_(free)(state, input);
}

void THNN_(Row2Col_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int inputWidth,
           int kW,
           int dW,
           int padW,
           int sW) {

  THNN_(Col2Row_updateOutput)(state, gradOutput, gradInput,
                              inputWidth,
                              kW, dW,
                              padW, sW);
}

#endif
