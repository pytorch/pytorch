#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Row2Col.c"
#else

static inline void THNN_(Row2Col_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
                         int kW, int dW,
                         int padW, int sW) {

  THArgCheck(kW > 0, 4,
             "kernel size should be greater than zero, but got kW: %d", kW);
  THArgCheck(dW > 0, 5,
             "dilation should be greater than zero, but got dW: %d", dW);
  THArgCheck(sW > 0, 7,
             "stride should be greater than zero, but got sW: %d", sW);

  int ndim = THTensor_(nDimension)(input);
  THNN_ARGCHECK(ndim == 2 || ndim == 3, 2, input,
                "2D or 3D input tensor expected but got: %s");

  int dim_batch = 0;
  if (ndim == 2) {
    dim_batch = -1;
  }
  int nInputPlane  = THTensor_(size)(input, dim_batch + 1);
  int inputWidth   = THTensor_(size)(input, dim_batch + 2);
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
           THNNState *state,
           THTensor *input,
           THTensor *output,
           int kW,
           int dW,
           int padW,
           int sW) {

  THNN_(Row2Col_shapeCheck)(state, input, NULL, kW, dW, padW, sW);

  input = THTensor_(newContiguous)(input);
  bool batched_input = true;
  if (input->nDimension == 2) {
    batched_input = false;
    THTensor_(resize3d)(input, 1, input->size[0], input->size[1]);
  }

  int batchSize    = THTensor_(size)(input, 0);
  int nInputPlane  = THTensor_(size)(input, 1);
  int inputWidth   = THTensor_(size)(input, 2);

  int outputWidth  = (inputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;
  int nOutputPlane = nInputPlane * kW;
  int outputLength = outputWidth;

  THTensor_(resize3d)(output, batchSize, nOutputPlane, outputLength);
  THTensor_(zero)(output);

  THTensor *input_n = THTensor_(new)();
  THTensor *output_n = THTensor_(new)();

  for (int elt = 0; elt < batchSize; elt++) {
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(output_n, output, 0, elt);

    THNN_(row2col)(
      THTensor_(data)(input_n),
      nInputPlane, inputWidth, kW, padW, sW,
      dW, THTensor_(data)(output_n));
  }

  THTensor_(free)(input_n);
  THTensor_(free)(output_n);

  if (!batched_input) {
    THTensor_(resize2d)(output, nOutputPlane, outputLength);
  }
  THTensor_(free)(input);
}

void THNN_(Row2Col_updateGradInput)(
           THNNState *state,
           THTensor *gradOutput,
           THTensor *gradInput,
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
