#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Col2Vol.c"
#else

static void THNN_(vol2col)(
  const real *data_vol, const int channels,
  const int depth, const int height, const int width,
  const int depth_col, const int height_col, const int width_col,
  const int kT, const int kH, const int kW,
  const int pT, const int pH, const int pW,
  const int dT, const int dH, const int dW,
  const int dilationT, const int dilationH, const int dilationW,
  real *data_col)
{
  int c, t, h, w;
  int channels_col = channels * kT * kH * kW;
  for (c = 0; c < channels_col; ++c)
  {
    int w_offset = c % kW;
    int h_offset = (c / kW) % kH;
    int t_offset = (c / kW / kH) % kT;
    int c_vol = c / kT / kH / kW;
    for (t = 0; t < depth_col; ++t)
    {
      for (h = 0; h < height_col; ++h)
      {
        for (w = 0; w < width_col; ++w)
        {
          int t_pad = t * dT - pT + t_offset * dilationT;
          int h_pad = h * dH - pH + h_offset * dilationH;
          int w_pad = w * dW - pW + w_offset * dilationW;
          if (t_pad >= 0 && t_pad < depth &&
              h_pad >= 0 && h_pad < height &&
              w_pad >= 0 && w_pad < width)
            data_col[((c * depth_col + t) * height_col + h) * width_col + w] =
              data_vol[((c_vol * depth + t_pad) * height + h_pad) * width + w_pad];
          else
            data_col[((c * depth_col + t) * height_col + h) * width_col + w] = 0;
        }
      }
    }
  }
}

static void THNN_(col2vol)(
  const real* data_col, const int channels,
  const int depth, const int height, const int width,
  const int out_depth, const int out_height, const int out_width,
  const int kT, const int kH, const int kW,
  const int pT, const int pH, const int pW,
  const int dT, const int dH, const int dW,
  const int dilationT, const int dilationH, const int dilationW,
  real* data_vol)
{
  int c, t, h, w;
  memset(data_vol, 0, sizeof(real) * depth * height * width * channels);
  int depth_col  = out_depth;
  int height_col = out_height;
  int width_col  = out_width;
  int channels_col = channels * kT * kH * kW;
  for (c = 0; c < channels_col; ++c)
  {
    int w_offset = c % kW;
    int h_offset = (c / kW) % kH;
    int t_offset = (c / kW / kH) % kT;
    int c_vol = c / kT / kH / kW;
    for (t = 0; t < depth_col; ++t)
    {
      for (h = 0; h < height_col; ++h)
      {
        for (w = 0; w < width_col; ++w)
        {
          int t_pad = t * dT - pT + t_offset * dilationT;
          int h_pad = h * dH - pH + h_offset * dilationH;
          int w_pad = w * dW - pW + w_offset * dilationW;
          if (t_pad >= 0 && t_pad < depth &&
              h_pad >= 0 && h_pad < height &&
              w_pad >= 0 && w_pad < width)
            data_vol[((c_vol * depth + t_pad) * height + h_pad) * width + w_pad] +=
              data_col[((c * depth_col + t) * height_col + h) * width_col + w];
        }
      }
    }
  }
}

static inline void THNN_(Col2Vol_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
                         int outputDepth, int outputHeight, int outputWidth,
                         int kT, int kH, int kW, int dT, int dH, int dW,
                         int padT, int padH, int padW, int sT, int sH, int sW) {

  THArgCheck(kW > 0 && kH > 0 && kT > 0, 6,
             "kernel size should be greater than zero, but got kT: %d kH: %d kW: %d", kT, kH, kW);
  THArgCheck(sW > 0 && sH > 0 && sT > 0, 14,
             "stride should be greater than zero, but got sT: %d sH: %d sW: %d", sT, sH, sW);
  THArgCheck(dW > 0 && dH > 0 && dT > 0, 9,
             "dilation should be greater than zero, but got dT: %d dH: %d dW: %d", dT, dH, dW);

  int ndim = THTensor_(nDimension)(input);
  THNN_ARGCHECK(ndim == 2 || ndim == 3, 2, input,
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
           THNNState *state,
           THTensor *input,
           THTensor *output,
           int outputDepth, int outputHeight, int outputWidth,
           int kT, int kH, int kW,
           int dT, int dH, int dW,
           int padT, int padH, int padW,
           int sT, int sH, int sW) {

  THNN_(Col2Vol_shapeCheck)(state, input, NULL, outputDepth, outputHeight, outputWidth,
                            kT, kH, kW, dT, dH, dW, padT, padH, padW, sT, sH, sW);

  bool batched_input = true;
  if (input->nDimension == 2) {
      // Force batch
      batched_input = false;
      THTensor_(resize3d)(input, 1, input->size[0], input->size[1]);
  }

  long batchSize = input->size[0];
  long nInputPlane = input->size[1];
  long nOutputPlane = nInputPlane / (kW * kH * kT);

  input = THTensor_(newContiguous)(input);

  THTensor_(resize5d)(output, batchSize, nOutputPlane, outputDepth, outputHeight, outputWidth);
  THTensor_(zero)(output);

  THTensor *input_n = THTensor_(new)();
  THTensor *output_n = THTensor_(new)();

  int depth_col = (outputDepth + 2 * padT - (dT * (kT - 1) + 1)) / sT + 1;
  int height_col = (outputHeight + 2 * padH - (dH * (kH - 1) + 1)) / sH + 1;
  int width_col = (outputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;

  for (int elt = 0; elt < batchSize; elt++) {
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(output_n, output, 0, elt);

    THNN_(col2vol)(
      THTensor_(data)(input_n),
      nOutputPlane,
      outputDepth, outputHeight, outputWidth,
      inputDepth, inputHeight, inputWidth,
      depth_col, height_col, width_col,
      kT, kH, kW,
      padT, padH, padW,
      sT, sH, sW,
      dT, dH, dW, THTensor_(data)(output_n));
  }

  THTensor_(free)(input_n);
  THTensor_(free)(output_n);

  if (!batched_input) {
      THTensor_(resize4d)(output, nOutputPlane, outputDepth, outputHeight, outputWidth);
  }
  THTensor_(free)(input);
}

void THNN_(Col2Vol_updateGradInput)(
           THNNState *state,
           THTensor *gradOutput,
           THTensor *gradInput,
           int kT, int kH, int kW,
           int dT, int dH, int dW,
           int padT, int padH, int padW,
           int sT, int sH, int sW) {

  THNN_(Vol2Col_updateOutput)(state, gradOutput, gradInput,
                              kT, kH, kW, dT, dH, dW, padT, padH, padW, sT, sH, sW);
}

#endif
