#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Col2Row.cu"
#else

static inline void THNN_(Col2Row_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         int outputWidth,
                         int kW, int dW,
                         int padW, int sW) {

  THArgCheck(kW > 0, 5,
             "kernel size should be greater than zero, but got kW: %d", kW);
  THArgCheck(sW > 0, 8,
             "stride should be greater than zero, but got sW: %d", sW);
  THArgCheck(dW > 0, 6,
             "dilation should be greater than zero, but got dW: %d", dW);

  int ndim = THCTensor_(nDimension)(state, input);
  THCUNN_argCheck(state, ndim == 2 || ndim == 3, 2, input,
                  "2D or 3D input tensor expected but got %s");

  int batch_dim = (ndim == 3) ? 0 : -1;
  long nInputPlane  = input->size[batch_dim + 1];
  long inputLength  = input->size[batch_dim + 2];

  long nOutputPlane = nInputPlane / kW;

  if (outputWidth < 1) {
    THError("Given input size: (%lld x %lld). "
            "Calculated output size: (%lld x %d). Output size is too small",
            (long long)nInputPlane, (long long)inputLength, (long long)nOutputPlane, outputWidth);
  }
}

void THNN_(Col2Row_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int outputWidth,
           int kW,
           int dW,
           int padW,
           int sW) {

  THCUNN_assertSameGPU(state, 2, input, output);

  THNN_(Col2Row_shapeCheck)(state, input, NULL, outputWidth,
                            kW, dW, padW, sW);

  bool batched_input = true;
  if (input->nDimension == 2) {
      // Force batch
      batched_input = false;
      THCTensor_(resize3d)(state, input, 1, input->size[0], input->size[1]);
  }

  long batchSize = input->size[0];
  long nInputPlane = input->size[1];
  long nOutputPlane = nInputPlane / kW;

  input = THCTensor_(newContiguous)(state, input);

  THCTensor_(resize3d)(state, output, batchSize, nOutputPlane, outputWidth);
  THCTensor_(zero)(state, output);

  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *output_n = THCTensor_(new)(state);

  int width_col = (outputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;

  for (int elt = 0; elt < batchSize; elt++) {
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);

    col2row<real, accreal>(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input_n),
      nOutputPlane,
      width_col,
      kW,
      padW,
      sW,
      dW, THCTensor_(data)(state, output_n));
  }

  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);

  if (!batched_input) {
      THCTensor_(resize2d)(state, output, nOutputPlane, outputWidth);
  }
  THCTensor_(free)(state, input);
}

void THNN_(Col2Row_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int kW,
           int dW,
           int padW,
           int sW) {

  THNN_(Row2Col_updateOutput)(state, gradOutput, gradInput,
                              kW, dW, padW, sW);

}

#endif
