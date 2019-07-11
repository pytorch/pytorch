#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/Im2Col.cu"
#else

#include <ATen/div_rtn.h>

static inline void THNN_(Im2Col_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         int64_t kH, int64_t kW, int64_t dH, int64_t dW,
                         int64_t padH, int64_t padW, int64_t sH, int64_t sW) {

  THArgCheck(kW > 0 && kH > 0, 4,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 6,
             "dilation should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THArgCheck(padW >= 0 && padH >= 0, 8,
             "padding should be non-negative, but got padH: %d padW: %d", padH, padW);
  THArgCheck(sW > 0 && sH > 0, 10,
             "stride should be greater than zero, but got sH: %d sW: %d", sH, sW);

  int64_t ndim = THCTensor_(nDimensionLegacyNoScalars)(state, input);
  THCUNN_argCheck(state, !input->is_empty() && (ndim == 3 || ndim == 4), 2, input,
                "Expected non-empty 3D or 4D input tensor, but got input of shape %s");

  int dim_batch = 0;
  if (ndim == 3) {
    dim_batch = -1;
  }
  int64_t nInputPlane  = THCTensor_(size)(state, input, dim_batch + 1);
  int64_t inputHeight  = THCTensor_(size)(state, input, dim_batch + 2);
  int64_t inputWidth   = THCTensor_(size)(state, input, dim_batch + 3);
  int64_t outputHeight = div_rtn<int64_t>(inputHeight + 2 * padH - (dH * (kH - 1) + 1),  sH) + 1;
  int64_t outputWidth  = div_rtn<int64_t>(inputWidth + 2 * padW - (dW * (kW - 1) + 1), sW) + 1;

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
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int64_t kH, int64_t kW,
           int64_t dH, int64_t dW,
           int64_t padH, int64_t padW,
           int64_t sH, int64_t sW) {

  THCUNN_assertSameGPU(state, 2, input, output);

  THNN_(Im2Col_shapeCheck)(state, input, NULL, kH, kW, dH, dW, padH, padW, sH, sW);

  input = THCTensor_(newContiguous)(state, input);
  bool batched_input = true;
  if (input->dim() == 3) {
    batched_input = false;
    THCTensor_(resize4d)(state, input, 1, input->size(0), input->size(1), input->size(2));
  }

  int64_t batchSize    = THCTensor_(size)(state, input, 0);
  int64_t nInputPlane  = THCTensor_(size)(state, input, 1);
  int64_t inputHeight  = THCTensor_(size)(state, input, 2);
  int64_t inputWidth   = THCTensor_(size)(state, input, 3);

  int64_t outputHeight = (inputHeight + 2 * padH - (dH * (kH - 1) + 1)) / sH + 1;
  int64_t outputWidth  = (inputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;
  int64_t nOutputPlane = nInputPlane * kW * kH;
  int64_t outputLength = outputHeight * outputWidth;

  THCTensor_(resize3d)(state, output, batchSize, nOutputPlane, outputLength);
  THCTensor_(zero)(state, output);

  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *output_n = THCTensor_(new)(state);

  for (int64_t elt = 0; elt < batchSize; elt++) {
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);

    im2col(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input_n),
      nInputPlane, inputHeight, inputWidth,
      outputHeight, outputWidth,
      kH, kW, padH, padW, sH, sW,
      dH, dW, THCTensor_(data)(state, output_n));
  }

  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);

  if (!batched_input) {
    THCTensor_(resize2d)(state, output, nOutputPlane, outputLength);
  }
  THCTensor_(free)(state, input);
}

void THNN_(Im2Col_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
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
