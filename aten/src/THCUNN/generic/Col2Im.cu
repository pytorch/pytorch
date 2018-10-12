#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/Col2Im.cu"
#else

#include <ATen/div_rtn.h>

static inline void THNN_(Col2Im_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         int64_t outputHeight, int64_t outputWidth,
                         int64_t kH, int64_t kW, int64_t dH, int64_t dW,
                         int64_t padH, int64_t padW, int64_t sH, int64_t sW) {

  THArgCheck(kW > 0 && kH > 0, 6,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(sW > 0 && sH > 0, 12,
             "stride should be greater than zero, but got sH: %d sW: %d", sH, sW);
  THArgCheck(dW > 0 && dH > 0, 8,
             "dilation should be greater than zero, but got dH: %d dW: %d", dH, dW);

  int64_t ndim = THCTensor_(nDimensionLegacyNoScalars)(state, input);
  THCUNN_argCheck(state, !input->is_empty() && (ndim == 2 || ndim == 3), 2, input,
                  "Expected non-empty 2D or 3D input tensor, but got input of shape %s");

  int batch_dim = (ndim == 3) ? 0 : -1;
  int64_t nInputPlane  = input->size(batch_dim + 1);

  if (nInputPlane % (kW * kH) != 0) {
    THError("Expected size of input's dimension 1 to be divisible by the "
            "product of kernel_size, but got input.size(1)=%lld and "
            "kernel_size=(%d, %d).", (long long) nInputPlane, kH, kW);
  }

  int64_t inputLength  = input->size(batch_dim + 2);
  int64_t nBlocksH = div_rtn<int64_t>(outputHeight + 2 * padH - dH * (kH - 1) - 1, sH) + 1;
  int64_t nBlocksW = div_rtn<int64_t>(outputWidth + 2 * padW - dW * (kW - 1) - 1, sW) + 1;

  if (inputLength != (nBlocksH * nBlocksW)) {
    THError("Given output_size=(%d, %d), kernel_size=(%d, %d), "
            "dilation=(%d, %d), padding=(%d, %d), stride=(%d, %d), expected "
            "size of input's dimension 2 to match the calculated number of "
            "sliding blocks %lld * %lld = %lld, but got input.size(2)=%lld.",
            outputHeight, outputWidth, kH, kW, dH, dW, padH, padW, sH, sW,
            (long long) nBlocksH, (long long) nBlocksW,
            (long long) (nBlocksH * nBlocksW), (long long) inputLength);
  }

  if (outputWidth < 1 || outputHeight < 1) {
    THError("Expected output spatial size to be positive, but got: output_size=(%d, %d).",
            outputHeight, outputWidth);
  }
}

void THNN_(Col2Im_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int64_t outputHeight, int64_t outputWidth,
           int64_t kH, int64_t kW,
           int64_t dH, int64_t dW,
           int64_t padH, int64_t padW,
           int64_t sH, int64_t sW) {

  THCUNN_assertSameGPU(state, 2, input, output);

  THNN_(Col2Im_shapeCheck)(state, input, NULL, outputHeight, outputWidth,
                           kH, kW, dH, dW, padH, padW, sH, sW);

  bool batched_input = true;
  if (input->dim() == 2) {
      // Force batch
      batched_input = false;
      THCTensor_(resize3d)(state, input, 1, input->size(0), input->size(1));
  }

  int64_t batchSize = input->size(0);
  int64_t nInputPlane = input->size(1);
  int64_t nOutputPlane = nInputPlane / (kW * kH);

  input = THCTensor_(newContiguous)(state, input);

  THCTensor_(resize4d)(state, output, batchSize, nOutputPlane, outputHeight, outputWidth);
  THCTensor_(zero)(state, output);

  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *output_n = THCTensor_(new)(state);

  int64_t height_col = (outputHeight + 2 * padH - (dH * (kH - 1) + 1)) / sH + 1;
  int64_t width_col = (outputWidth + 2 * padW - (dW * (kW - 1) + 1)) / sW + 1;

  for (int64_t elt = 0; elt < batchSize; elt++) {
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);

    col2im<scalar_t, accreal>(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input_n),
      nOutputPlane,
      outputHeight, outputWidth,
      height_col, width_col,
      kH, kW,
      padH, padW,
      sH, sW,
      dH, dW, THCTensor_(data)(state, output_n));
  }

  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);

  if (!batched_input) {
      THCTensor_(resize3d)(state, output, nOutputPlane, outputHeight, outputWidth);
  }
  THCTensor_(free)(state, input);
}

void THNN_(Col2Im_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int64_t kH, int64_t kW,
           int64_t dH, int64_t dW,
           int64_t padH, int64_t padW,
           int64_t sH, int64_t sW) {

  THNN_(Im2Col_updateOutput)(state, gradOutput, gradInput,
                             kH, kW, dH, dW, padH, padW, sH, sW);

}

#endif
