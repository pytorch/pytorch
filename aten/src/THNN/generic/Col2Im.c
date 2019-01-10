#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Col2Im.c"
#else

// Note [im2col/col2im output padding]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Our implementations of im2col and col2im take both the input height/width as
// well as a seemingly redundant output height/width.  In principle, you could
// compute the output height/width by using the convolution shape formulas.  So,
// what's up with that?
//
// The trouble arises when one runs the backward of a transposed convolution
// with output_padding >= stride.  (BTW, output_padding is known as adj inside
// THNN.) Let's consider a simple case where we have kernel=2, dilation=2,
// stride=1, output_padding=1 for a 4x4 input:
//
// Input:  X
//
// Output: X.X.
//         ....
//         X.X.
//         ....
//
// If we compute backwards of output with a standard convolution on the output
// with the same parameters, we would end up with a 2x2 grad_input (because you
// can slide the stencil over to the right once and down once).  But that is all
// out-of-bounds if you're computing backwards for a 1x1 input.
//
// "Now Edward," you might say, "the real problem is that you set output_padding
// >= stride, surely an error should have been raised in this case."  To
// understand why it is useful to handle this case, we have to understand how we
// compute the weight gradient of a convolution.  Suppose we have a convolution
// with kernel=2, stride=2 on a 5x5 input.  Let us see all the contributions of
// weight[0][0] (which we have labeled w) in the output:
//
// Input:  a.b..  Weight: w.
//         .....          ..
//         c.d..
//         .....
//         .....
//
// Output: [ aw+...  bw+... ]
//         [ cw+...  dw+... ]
//
// From this diagram, it easy to see that we can compute the weight gradient
// by performing a *dilated* convolution between the input and the
// output gradients with kernel=2, dilation=2, stride=1.  But there's a rub: if
// we do a dilated convolution directly, we'll end up with a 3x3 weight
// gradient, when we clearly wanted a 2x2.  So how do we avoid going out
// of bounds?  We could add a notion of 'output_padding' for non-transposed
// convolution, but another simple and effective fix is to just accept
// the desired output size directly, and compute only within those bounds.
//
//
// ALSO do vol2col

static void THNN_(im2col)(const real* data_im, const int channels,
      const int height, const int width,
      const int output_height, const int output_width,
      const int kernel_h, const int kernel_w,
      const int pad_h, const int pad_w,
      const int stride_h, const int stride_w,
      const int dilation_h, const int dilation_w,
      real* data_col) {
  const int height_col = output_height;
  const int width_col = output_width;
  const int channels_col = channels * kernel_h * kernel_w;
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    int w_offset = c_col % kernel_w;
    int h_offset = (c_col / kernel_w) % kernel_h;
    int c_im = c_col / kernel_h / kernel_w;
    for (int h_col = 0; h_col < height_col; ++h_col) {
      for (int w_col = 0; w_col < width_col; ++w_col) {
        int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
        int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
        data_col[(c_col * height_col + h_col) * width_col + w_col] =
          (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
          data_im[(c_im * height + h_im) * width + w_im] : 0;
      }
    }
  }
}

static void THNN_(col2im)(const real* data_col, const int channels,
      const int height, const int width,
      const int output_height, const int output_width,
      const int kernel_h, const int kernel_w,
      const int pad_h, const int pad_w,
      const int stride_h, const int stride_w,
      const int dilation_h, const int dilation_w,
      real* data_im) {
  memset(data_im, 0, sizeof(real) * height * width * channels);
  const int height_col = output_height;
  const int width_col = output_width;
  const int channels_col = channels * kernel_h * kernel_w;
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    int w_offset = c_col % kernel_w;
    int h_offset = (c_col / kernel_w) % kernel_h;
    int c_im = c_col / kernel_h / kernel_w;
    for (int h_col = 0; h_col < height_col; ++h_col) {
      for (int w_col = 0; w_col < width_col; ++w_col) {
        int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
        int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
        if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
          data_im[(c_im * height + h_im) * width + w_im] +=
            data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
  }
}

static inline void THNN_(Col2Im_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
                         int outputHeight, int outputWidth,
                         int kH, int kW, int dH, int dW,
                         int padH, int padW, int sH, int sW) {

  THArgCheck(kW > 0 && kH > 0, 6,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(sW > 0 && sH > 0, 12,
             "stride should be greater than zero, but got sH: %d sW: %d", sH, sW);
  THArgCheck(dW > 0 && dH > 0, 8,
             "dilation should be greater than zero, but got dH: %d dW: %d", dH, dW);

  int ndim = THTensor_(nDimension)(input);
  THNN_ARGCHECK(ndim == 2 || ndim == 3, 2, input,
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
           THNNState *state,
           THTensor *input,
           THTensor *output,
           int outputHeight, int outputWidth,
           int kH, int kW,
           int dH, int dW,
           int padH, int padW,
           int sH, int sW) {

  THNN_(Col2Im_shapeCheck)(state, input, NULL, outputHeight, outputWidth,
                           kH, kW, dH, dW, padH, padW, sH, sW);

  bool batched_input = true;
  if (input->nDimension == 2) {
      // Force batch
      batched_input = false;
      THTensor_(resize3d)(input, 1, input->size[0], input->size[1]);
  }

  long batchSize = input->size[0];
  long nInputPlane = input->size[1];
  long nOutputPlane = nInputPlane / (kW * kH);

  input = THTensor_(newContiguous)(input);

  THTensor_(resize4d)(output, batchSize, nOutputPlane, outputHeight, outputWidth);
  THTensor_(zero)(output);

  THTensor *input_n = THTensor_(new)();
  THTensor *output_n = THTensor_(new)();

  int height_col = (outputHeight + 2 * padH - (dH * (kH - 1) + 1)) / sH + 1;
  int width_col = (outputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;

  for (int elt = 0; elt < batchSize; elt++) {
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(output_n, output, 0, elt);

    THNN_(col2im)(
      THTensor_(data)(input_n),
      nOutputPlane,
      outputHeight, outputWidth,
      height_col, width_col,
      kH, kW,
      padH, padW,
      sH, sW,
      dH, dW, THTensor_(data)(output_n));
  }

  THTensor_(free)(input_n);
  THTensor_(free)(output_n);

  if (!batched_input) {
      THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);
  }
  THTensor_(free)(input);
}

void THNN_(Col2Im_updateGradInput)(
           THNNState *state,
           THTensor *gradOutput,
           THTensor *gradInput,
           int kH, int kW,
           int dH, int dW,
           int padH, int padW,
           int sH, int sW) {

  THNN_(Im2Col_updateOutput)(state, gradOutput, gradInput,
                             kH, kW, dH, dW, padH, padW, sH, sW);
}

#endif
