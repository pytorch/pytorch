#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/Col2Im.c"
#else

#include <ATen/div_rtn.h>

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

static void THNN_(im2col)(const scalar_t* data_im, const int64_t channels,
      const int64_t height, const int64_t width,
      const int64_t output_height, const int64_t output_width,
      const int64_t kernel_h, const int64_t kernel_w,
      const int64_t pad_h, const int64_t pad_w,
      const int64_t stride_h, const int64_t stride_w,
      const int64_t dilation_h, const int64_t dilation_w,
      scalar_t* data_col) {
  const int64_t height_col = output_height;
  const int64_t width_col = output_width;
  const int64_t channels_col = channels * kernel_h * kernel_w;
  for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
    int64_t w_offset = c_col % kernel_w;
    int64_t h_offset = (c_col / kernel_w) % kernel_h;
    int64_t c_im = c_col / kernel_h / kernel_w;
    for (int64_t h_col = 0; h_col < height_col; ++h_col) {
      int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
      for (int64_t w_col = 0; w_col < width_col; ++w_col) {
        int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
        data_col[(c_col * height_col + h_col) * width_col + w_col] =
          (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
          data_im[(c_im * height + h_im) * width + w_im] : 0;
      }
    }
  }
}

static void THNN_(col2im)(const scalar_t* data_col, const int64_t channels,
      const int64_t height, const int64_t width,
      const int64_t output_height, const int64_t output_width,
      const int64_t kernel_h, const int64_t kernel_w,
      const int64_t pad_h, const int64_t pad_w,
      const int64_t stride_h, const int64_t stride_w,
      const int64_t dilation_h, const int64_t dilation_w,
      scalar_t* data_im) {
  memset(data_im, 0, sizeof(scalar_t) * height * width * channels);
  const int64_t height_col = output_height;
  const int64_t width_col = output_width;
  const int64_t channels_col = channels * kernel_h * kernel_w;
  for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
    int64_t w_offset = c_col % kernel_w;
    int64_t h_offset = (c_col / kernel_w) % kernel_h;
    int64_t c_im = c_col / kernel_h / kernel_w;
    for (int64_t h_col = 0; h_col < height_col; ++h_col) {
      int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
      for (int64_t w_col = 0; w_col < width_col; ++w_col) {
        int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
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
                         int64_t outputHeight, int64_t outputWidth,
                         int64_t kH, int64_t kW, int64_t dilationH, int64_t dilationW,
                         int64_t padH, int64_t padW, int64_t dH, int64_t dW) {

  THArgCheck(kW > 0 && kH > 0, 6,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 12,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THArgCheck(dilationW > 0 && dilationH > 0, 8,
             "dilation should be greater than zero, but got dilationH: %d dilationW: %d", dilationH, dilationW);

  int64_t ndim = THTensor_(nDimensionLegacyNoScalars)(input);
  THNN_ARGCHECK(!input->is_empty() && (ndim == 2 || ndim == 3), 2, input,
                "Expected non-empty 2D or 3D input tensor, but got input of shape %s");

  int64_t batch_dim = (ndim == 3) ? 0 : -1;
  int64_t nInputPlane  = input->size(batch_dim + 1);

  if (nInputPlane % (kW * kH) != 0) {
    THError("Expected size of input's dimension 1 to be divisible by the "
            "product of kernel_size, but got input.size(1)=%lld and "
            "kernel_size=(%d, %d).", (long long) nInputPlane, kH, kW);
  }

  int64_t inputLength  = input->size(batch_dim + 2);
  int64_t nBlocksH = div_rtn<int64_t>(outputHeight + 2 * padH - dilationH * (kH - 1) - 1, dH) + 1;
  int64_t nBlocksW = div_rtn<int64_t>(outputWidth + 2 * padW - dilationW * (kW - 1) - 1, dW) + 1;

  if (inputLength != (nBlocksH * nBlocksW)) {
    THError("Given output_size=(%d, %d), kernel_size=(%d, %d), "
            "dilation=(%d, %d), padding=(%d, %d), stride=(%d, %d), expected "
            "size of input's dimension 2 to match the calculated number of "
            "sliding blocks %lld * %lld = %lld, but got input.size(2)=%lld.",
            outputHeight, outputWidth, kH, kW, dilationH, dilationW, padH, padW, dH, dW,
            (long long) nBlocksH, (long long) nBlocksW,
            (long long) (nBlocksH * nBlocksW), (long long) inputLength);
  }

  if (outputWidth < 1 || outputHeight < 1) {
    THError("Expected output spatial size to be positive, but got: output_size=(%d, %d).",
            outputHeight, outputWidth);
  }
}

void THNN_(Col2Im_updateOutput)(
           THNNState *state,
           THTensor *input,
           THTensor *output,
           int64_t outputHeight, int64_t outputWidth,
           int64_t kH, int64_t kW,
           int64_t dilationH, int64_t dilationW,
           int64_t padH, int64_t padW,
           int64_t dH, int64_t dW) {

  THNN_(Col2Im_shapeCheck)(state, input, NULL, outputHeight, outputWidth,
                           kH, kW, dilationH, dilationW, padH, padW, dH, dW);

  bool batched_input = true;
  if (input->dim() == 2) {
      // Force batch
      batched_input = false;
      THTensor_(resize3d)(input, 1, input->size(0), input->size(1));
  }

  long batchSize = input->size(0);
  long nInputPlane = input->size(1);
  long nOutputPlane = nInputPlane / (kW * kH);

  input = THTensor_(newContiguous)(input);

  THTensor_(resize4d)(output, batchSize, nOutputPlane, outputHeight, outputWidth);
  THTensor_(zero)(output);

  THTensor *input_n = THTensor_(new)();
  THTensor *output_n = THTensor_(new)();

  int64_t height_col = (outputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  int64_t width_col = (outputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  for (int64_t elt = 0; elt < batchSize; elt++) {
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(output_n, output, 0, elt);

    THNN_(col2im)(
      input_n->data<scalar_t>(),
      nOutputPlane,
      outputHeight, outputWidth,
      height_col, width_col,
      kH, kW,
      padH, padW,
      dH, dW,
      dilationH, dilationW, output_n->data<scalar_t>());
  }

  c10::raw::intrusive_ptr::decref(input_n);
  c10::raw::intrusive_ptr::decref(output_n);

  if (!batched_input) {
      THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);
  }
  c10::raw::intrusive_ptr::decref(input);
}

void THNN_(Col2Im_updateGradInput)(
           THNNState *state,
           THTensor *gradOutput,
           THTensor *gradInput,
           int64_t kH, int64_t kW,
           int64_t dilationH, int64_t dilationW,
           int64_t padH, int64_t padW,
           int64_t dH, int64_t dW) {

  THNN_(Im2Col_updateOutput)(state, gradOutput, gradInput,
                             kH, kW, dilationH, dilationW, padH, padW, dH, dW);
}

#endif
