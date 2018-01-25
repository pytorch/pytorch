#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Col2Row.c"
#else

static void THNN_(row2col)(const real* data_im, const int channels,
      const int width, const int kernel_w,
      const int pad_w,
      const int stride_w,
      const int dilation_w,
      real* data_col) {
  const int width_col = (width + 2 * pad_w -
                         (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channels_col = channels * kernel_w;
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    int w_offset = c_col % kernel_w;
    int c_im = c_col / kernel_w;
    for (int w_col = 0; w_col < width_col; ++w_col) {
      int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
      data_col[c_col * width_col + w_col] =
        (w_im >= 0 && w_im < width) ?
        data_im[c_im * width + w_im] : 0;
    }
  }
}

static void THNN_(col2row)(const real* data_col, const int channels,
      const int width,
      const int output_width,
      const int kernel_w,
      const int pad_w,
      const int stride_w,
      const int dilation_w,
      real* data_im) {
  memset(data_im, 0, sizeof(real) * width * channels);
  const int width_col = output_width;
  const int channels_col = channels * kernel_w;
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    int w_offset = c_col % kernel_w;
    int c_im = c_col / kernel_w;
    for (int w_col = 0; w_col < width_col; ++w_col) {
      int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
      if (w_im >= 0 && w_im < width)
        data_im[c_im * width + w_im] +=
          data_col[c_col * width_col + w_col];
    }
  }
}

static inline void THNN_(Col2Row_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
                         int outputWidth,
                         int kW, int dW,
                         int padW, int sW) {

  THArgCheck(kW > 0, 6,
             "kernel size should be greater than zero, but got kW: %d", kW);
  THArgCheck(sW > 0, 9,
             "stride should be greater than zero, but got sW: %d", sW);
  THArgCheck(dW > 0, 7,
             "dilation should be greater than zero, but got dW: %d", dW);

  int ndim = THTensor_(nDimension)(input);
  THNN_ARGCHECK(ndim == 2 || ndim == 3, 2, input,
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
           THNNState *state,
           THTensor *input,
           THTensor *output,
           int outputWidth,
           int kW,
           int dW,
           int padW,
           int sW) {

  THNN_(Col2Row_shapeCheck)(state, input, NULL, outputWidth,
                            kW, dW, padW, sW);

  bool batched_input = true;
  if (input->nDimension == 2) {
      // Force batch
      batched_input = false;
      THTensor_(resize3d)(input, 1, input->size[0], input->size[1]);
  }

  long batchSize = input->size[0];
  long nInputPlane = input->size[1];
  long nOutputPlane = nInputPlane / kW;

  input = THTensor_(newContiguous)(input);

  THTensor_(resize3d)(output, batchSize, nOutputPlane, outputWidth);
  THTensor_(zero)(output);

  THTensor *input_n = THTensor_(new)();
  THTensor *output_n = THTensor_(new)();

  int width_col = (outputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;

  for (int elt = 0; elt < batchSize; elt++) {
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(output_n, output, 0, elt);

    THNN_(col2row)(
      THTensor_(data)(input_n),
      nOutputPlane,
      outputWidth,
      width_col,
      kW,
      padW,
      sW,
      dW, THTensor_(data)(output_n));
  }

  THTensor_(free)(input_n);
  THTensor_(free)(output_n);

  if (!batched_input) {
      THTensor_(resize2d)(output, nOutputPlane, outputWidth);
  }
  THTensor_(free)(input);
}

void THNN_(Col2Row_updateGradInput)(
           THNNState *state,
           THTensor *gradOutput,
           THTensor *gradInput,
           int kW,
           int dW,
           int padW,
           int sW) {

  THNN_(Row2Col_updateOutput)(state, gradOutput, gradInput,
                              kW, dW, padW, sW);
}

#endif
