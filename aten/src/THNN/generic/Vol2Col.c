#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Vol2Col.c"
#else

static inline void THNN_(Vol2Col_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
                         int kT, int kH, int kW, int dT, int dH, int dW,
                         int padT, int padH, int padW, int sT, int sH, int sW) {

  THArgCheck(kW > 0 && kH > 0 && kT > 0, 4,
             "kernel size should be greater than zero, but got kT: %d kH: %d kW: %d", kT, kH, kW);
  THArgCheck(dW > 0 && dH > 0 && dT > 0, 7,
             "dilation should be greater than zero, but got dT: %d dH: %d dW: %d", dT, dH, dW);
  THArgCheck(sW > 0 && sH > 0 && sT > 0, 12,
             "stride should be greater than zero, but got sT: %d sH: %d sW: %d", sT, sH, sW);

  int ndim = THTensor_(nDimension)(input);
  THNN_ARGCHECK(ndim == 4 || ndim == 5, 2, input,
                "4D or 5D input tensor expected but got: %s");

  int dim_batch = 0;
  if (ndim == 4) {
    dim_batch = -1;
  }
  int nInputPlane  = THTensor_(size)(input, dim_batch + 1);
  int inputDepth   = THTensor_(size)(input, dim_batch + 2);
  int inputHeight  = THTensor_(size)(input, dim_batch + 3);
  int inputWidth   = THTensor_(size)(input, dim_batch + 4);
  int outputDepth  = (inputDepth + 2 * padT - (dT * (kT - 1) + 1)) / sT + 1;
  int outputHeight = (inputHeight + 2 * padH - (dH * (kH - 1) + 1)) / sH + 1;
  int outputWidth  = (inputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;
  int nOutputPlane = nInputPlane * kT * kW * kH;
  int outputLength = outputDepth * outputHeight * outputWidth;

  if (outputWidth < 1 || outputHeight < 1 || outputDepth < 1) {
    THError("Given input size: (%d x %d x %d x %d). "
            "Calculated output size: (%d x %d). Output size is too small",
            nInputPlane, inputDepth, inputHeight, inputWidth, nOutputPlane, outputLength);
  }
}

void THNN_(Vol2Col_updateOutput)(
           THNNState *state,
           THTensor *input,
           THTensor *output,
           int kT, int kH, int kW,
           int dT, int dH, int dW,
           int padT, int padH, int padW,
           int sT, int sH, int sW) {

  THNN_(Vol2Col_shapeCheck)(state, input, NULL, kT, kH, kW, dT, dH, dW, padT, padH, padW, sT, sH, sW);

  input = THTensor_(newContiguous)(input);
  bool batched_input = true;
  if (input->nDimension == 4) {
    batched_input = false;
    THTensor_(resize5d)(input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
  }

  int batchSize    = THTensor_(size)(input, 0);
  int nInputPlane  = THTensor_(size)(input, 1);
  int inputDepth   = THTensor_(size)(input, 2);
  int inputHeight  = THTensor_(size)(input, 3);
  int inputWidth   = THTensor_(size)(input, 4);

  int outputDepth  = (inputDepth + 2 * padT - (dT * (kT - 1) + 1)) / sT + 1;
  int outputHeight = (inputHeight + 2 * padH - (dH * (kH - 1) + 1)) / sH + 1;
  int outputWidth  = (inputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;
  int nOutputPlane = nInputPlane * kW * kH * kT;
  int outputLength = outputDepth * outputHeight * outputWidth;

  THTensor_(resize3d)(output, batchSize, nOutputPlane, outputLength);
  THTensor_(zero)(output);

  THTensor *input_n = THTensor_(new)();
  THTensor *output_n = THTensor_(new)();

  for (int elt = 0; elt < batchSize; elt++) {
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(output_n, output, 0, elt);

    THNN_(vol2col)(
      THTensor_(data)(input_n),
      nInputPlane, inputDepth, inputHeight, inputWidth,
      outputDepth, outputHeight, outputWidth,
      kT, kH, kW, padT, padH, padW, sT, sH, sW,
      dT, dH, dW, THTensor_(data)(output_n));
  }

  THTensor_(free)(input_n);
  THTensor_(free)(output_n);

  if (!batched_input) {
    THTensor_(resize2d)(output, nOutputPlane, outputLength);
  }
  THTensor_(free)(input);
}

void THNN_(Vol2Col_updateGradInput)(
           THNNState *state,
           THTensor *gradOutput,
           THTensor *gradInput,
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
