#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Im2Col.c"
#else

#include <ATen/div_rtn.h>

static inline void THNN_(Im2Col_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
                         int64_t kH, int64_t kW, int64_t dH, int64_t dW,
                         int64_t padH, int64_t padW, int64_t sH, int64_t sW) {

  THArgCheck(kW > 0 && kH > 0, 4,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 6,
             "dilation should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THArgCheck(sW > 0 && sH > 0, 10,
             "stride should be greater than zero, but got sH: %d sW: %d", sH, sW);

  int64_t ndim = THTensor_(nDimensionLegacyNoScalars)(input);
  THNN_ARGCHECK(!input->is_empty() && (ndim == 3 || ndim == 4), 2, input,
                "Expected non-empty 3D or 4D input tensor, but got input of shape %s");

  int64_t dim_batch = 0;
  if (ndim == 3) {
    dim_batch = -1;
  }
  int64_t nInputPlane  = THTensor_(size)(input, dim_batch + 1);
  int64_t inputHeight  = THTensor_(size)(input, dim_batch + 2);
  int64_t inputWidth   = THTensor_(size)(input, dim_batch + 3);
  int64_t outputHeight = div_rtn<int64_t>(inputHeight + 2 * padH - (dH * (kH - 1) + 1), sH) + 1;
  int64_t outputWidth  = div_rtn<int64_t>(inputWidth + 2 * padW - (dW * (kW - 1) + 1), sW) + 1;
  int64_t nOutputPlane = nInputPlane * kW * kH;
  int64_t outputLength = outputHeight * outputWidth;

  if (outputHeight < 1 || outputWidth < 1) {
    THError("Given input with spatial size (%d, %d), kernel_size=(%d, %d), "
            "dilation=(%d, %d), padding=(%d, %d), calculated "
            "shape of the array of sliding blocks as (%d, %d), which is "
            "too small (non-positive).",
            inputHeight, inputHeight, kH, kW, dH, dW, padH, padW,
            outputHeight, outputWidth);
  }
}

void THNN_(Im2Col_updateOutput)(
           THNNState *state,
           THTensor *input,
           THTensor *output,
           int64_t kH, int64_t kW,
           int64_t dH, int64_t dW,
           int64_t padH, int64_t padW,
           int64_t sH, int64_t sW) {

  THNN_(Im2Col_shapeCheck)(state, input, NULL, kH, kW, dH, dW, padH, padW, sH, sW);

  input = THTensor_(newContiguous)(input);
  bool batched_input = true;
  if (input->dim() == 3) {
    batched_input = false;
    THTensor_(resize4d)(input, 1, input->size(0), input->size(1), input->size(2));
  }

  int64_t batchSize    = THTensor_(size)(input, 0);
  int64_t nInputPlane  = THTensor_(size)(input, 1);
  int64_t inputHeight  = THTensor_(size)(input, 2);
  int64_t inputWidth   = THTensor_(size)(input, 3);

  int64_t outputHeight = (inputHeight + 2 * padH - (dH * (kH - 1) + 1)) / sH + 1;
  int64_t outputWidth  = (inputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;
  int64_t nOutputPlane = nInputPlane * kW * kH;
  int64_t outputLength = outputHeight * outputWidth;

  THTensor_(resize3d)(output, batchSize, nOutputPlane, outputLength);
  THTensor_(zero)(output);

  THTensor *input_n = THTensor_(new)();
  THTensor *output_n = THTensor_(new)();

  for (int64_t elt = 0; elt < batchSize; elt++) {
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(output_n, output, 0, elt);

    THNN_(im2col)(
      input_n->data<scalar_t>(),
      nInputPlane,
      inputHeight, inputWidth,
      outputHeight, outputWidth,
      kH, kW, padH, padW, sH, sW,
      dH, dW, output_n->data<scalar_t>());
  }

  c10::raw::intrusive_ptr::decref(input_n);
  c10::raw::intrusive_ptr::decref(output_n);

  if (!batched_input) {
    THTensor_(resize2d)(output, nOutputPlane, outputLength);
  }
  c10::raw::intrusive_ptr::decref(input);
}

void THNN_(Im2Col_updateGradInput)(
           THNNState *state,
           THTensor *gradOutput,
           THTensor *gradInput,
           int64_t inputHeight, int64_t inputWidth,
           int64_t kH, int64_t kW,
           int64_t dH, int64_t dW,
           int64_t padH, int64_t padW,
           int64_t sH, int64_t sW) {


  THNN_(Col2Im_updateOutput)(state, gradOutput, gradInput,
                             inputHeight, inputWidth,
                             kH, kW, dH, dW,
                             padH, padW, sH, sW);
}


#endif
