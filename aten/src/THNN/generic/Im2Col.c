#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Im2Col.c"
#else

static inline void THNN_(Im2Col_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
                         int kH, int kW, int dH, int dW,
                         int padH, int padW, int sH, int sW) {

  THArgCheck(kW > 0 && kH > 0, 4,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 6,
             "dilation should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THArgCheck(sW > 0 && sH > 0, 10,
             "stride should be greater than zero, but got sH: %d sW: %d", sH, sW);

  int ndim = THTensor_(nDimension)(input);
  THNN_ARGCHECK(ndim == 3 || ndim == 4, 2, input,
                "3D or 4D input tensor expected but got: %s");

  int dim_batch = 0;
  if (ndim == 3) {
    dim_batch = -1;
  }
  int nInputPlane  = THTensor_(size)(input, dim_batch + 1);
  int inputHeight  = THTensor_(size)(input, dim_batch + 2);
  int inputWidth   = THTensor_(size)(input, dim_batch + 3);
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
           THNNState *state,
           THTensor *input,
           THTensor *output,
           int kH, int kW,
           int dH, int dW,
           int padH, int padW,
           int sH, int sW) {

  THNN_(Im2Col_shapeCheck)(state, input, NULL, kH, kW, dH, dW, padH, padW, sH, sW);

  input = THTensor_(newContiguous)(input);
  bool batched_input = true;
  if (input->nDimension == 3) {
    batched_input = false;
    THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
  }

  int batchSize    = THTensor_(size)(input, 0);
  int nInputPlane  = THTensor_(size)(input, 1);
  int inputHeight  = THTensor_(size)(input, 2);
  int inputWidth   = THTensor_(size)(input, 3);

  int outputHeight = (inputHeight + 2 * padH - (dH * (kH - 1) + 1)) / sH + 1;
  int outputWidth  = (inputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;
  int nOutputPlane = nInputPlane * kW * kH;
  int outputLength = outputHeight * outputWidth;

  THTensor_(resize3d)(output, batchSize, nOutputPlane, outputLength);
  THTensor_(zero)(output);

  THTensor *input_n = THTensor_(new)();
  THTensor *output_n = THTensor_(new)();

  for (int elt = 0; elt < batchSize; elt++) {
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(output_n, output, 0, elt);

    THNN_(im2col)(
      THTensor_(data)(input_n),
      nInputPlane,
      inputHeight, inputWidth,
      outputHeight, outputWidth,
      kH, kW, padH, padW, sH, sW,
      dH, dW, THTensor_(data)(output_n));
  }

  THTensor_(free)(input_n);
  THTensor_(free)(output_n);

  if (!batched_input) {
    THTensor_(resize2d)(output, nOutputPlane, outputLength);
  }
  THTensor_(free)(input);
}

void THNN_(Im2Col_updateGradInput)(
           THNNState *state,
           THTensor *gradOutput,
           THTensor *gradInput,
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
