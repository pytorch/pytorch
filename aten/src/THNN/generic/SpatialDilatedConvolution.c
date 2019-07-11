#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/SpatialDilatedConvolution.c"
#else

#include <ATen/div_rtn.h>

static inline void THNN_(SpatialDilatedConvolution_shapeCheck)(
        THTensor *input, THTensor *gradOutput,
        THTensor *weight, THTensor *bias,
        int kH, int kW, int dH, int dW, int padH, int padW,
        int dilationH, int dilationW, int weight_nullable) {
  THArgCheck(kW > 0 && kH > 0, 9,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THArgCheck(dilationW > 0 && dilationH > 0, 15,
             "dilation should be greater than zero, but got dilationH: %d, dilationW: %d",
             dilationH, dilationW);

  if (weight != NULL) {
    THNN_ARGCHECK(!weight->is_empty() && weight->dim() == 4, 4, weight,
                  "non-empty 4D weight tensor (nOutputPlane, nInputPlane, kH, kW) expected, "
                  "but got: %s");
    if (bias != NULL) {
      THNN_CHECK_DIM_SIZE(bias, 1, 0, weight->size(0));
    }
  } else if (!weight_nullable) {
    THError("weight tensor is expected to be non-nullable");
  }

  int ndim = input->dim();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THNN_ARGCHECK(!input->is_empty() && (ndim == 3 || ndim == 4), 2, input,
                "non-empty 3D or 4D input tensor expected but got: %s");

  int64_t inputHeight  = input->size(dimh);
  int64_t inputWidth   = input->size(dimw);

  int64_t outputHeight = div_rtn<int64_t>(inputHeight + 2*padH - (dilationH * (kH - 1) + 1), dH) + 1;
  int64_t outputWidth  = div_rtn<int64_t>(inputWidth + 2*padW - (dilationW * (kW - 1) + 1), dW) + 1;

  if (outputWidth < 1 || outputHeight < 1) {
    THError("Given input size per channel: (%ld x %ld). "
      "Calculated output size per channel: (%ld x %ld). Output size is too small",
      inputHeight, inputWidth, outputHeight, outputWidth);
  }

  if (weight != NULL) {
    int64_t nInputPlane = weight->size(1);
    THNN_CHECK_DIM_SIZE(input, ndim, dimf, nInputPlane);
  }

  if (gradOutput != NULL) {
    if (weight != NULL) {
      int64_t nOutputPlane = weight->size(0);
      THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    } else if (bias != NULL) {
      int64_t nOutputPlane = THTensor_sizeLegacyNoScalars(bias, 0);
      THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    }
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
}

void THNN_(SpatialDilatedConvolution_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    THTensor *weight,
    THTensor *bias,
    THTensor *columns,
    THTensor *ones,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH)
{

  THNN_(SpatialDilatedConvolution_shapeCheck)
    (input, NULL, weight, bias, kH, kW, dH, dW, padH, padW,
     dilationH, dilationW, 0);

  // Params:
  int nInputPlane = weight->size(1);
  int nOutputPlane = weight->size(0);

  input = THTensor_(newContiguous)(input);
  weight = THTensor_(newContiguous)(weight);
  THArgCheck(THTensor_(isContiguous)(columns), 5, "columns needs to be contiguous");
  if (bias) {
    bias = THTensor_(newContiguous)(bias);
    THArgCheck(THTensor_(isContiguous)(ones), 6, "ones needs to be contiguous");
  }
  int is_batch = 1;
  if (input->dim() == 3) {
    // Force batch
    is_batch = 0;
    THTensor_(resize4d)(input, 1, input->size(0), input->size(1), input->size(2));
  }
  int64_t inputWidth   = input->size(3);
  int64_t inputHeight  = input->size(2);
  int64_t outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  // Batch size + input planes
  int64_t batchSize = input->size(0);

  // Resize output
  THTensor_(resize4d)(output, batchSize, nOutputPlane, outputHeight, outputWidth);
  THTensor_(zero)(output);

  // Resize temporary columns
  THTensor_(resize2d)(columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (!THTensor_(isContiguous)(ones) || ones->dim() != 2 ||
      ones->size(0)*ones->size(1) < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THTensor_(resize2d)(ones, outputHeight, outputWidth);
    THTensor_(fill)(ones, 1);
  }

  // Helpers
  THTensor *input_n = THTensor_(new)();
  THTensor *output_n = THTensor_(new)();

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(output_n, output, 0, elt);

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    int64_t m_ = nOutputPlane;
    int64_t n_ = outputHeight * outputWidth;
    int64_t k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    if (bias) {
      THBlas_(gemm)(
        't', 'n',
        n_, m_, k_,
        1,
        ones->data<scalar_t>(), k_,
        bias->data<scalar_t>(), k_,
        0,
        output_n->data<scalar_t>(), n_
      );
    } else {
      THTensor_(zero)(output_n);
    }

    // Extract columns:
    THNN_(im2col)(
      input_n->data<scalar_t>(),
      nInputPlane, inputHeight, inputWidth,
      outputHeight, outputWidth,
      kH, kW, padH, padW, dH, dW,
      dilationH, dilationW,
      columns->data<scalar_t>()
    );

    // M,N,K are dims of matrix A and B
    int64_t m = nOutputPlane;
    int64_t n = columns->size(1);
    int64_t k = nInputPlane*kH*kW;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THBlas_(gemm)(
      'n', 'n',
      n, m, k,
      1,
      columns->data<scalar_t>(), n,
      weight->data<scalar_t>(), k,
      1,
      output_n->data<scalar_t>(), n
    );
  }

  // Free
  c10::raw::intrusive_ptr::decref(input_n);
  c10::raw::intrusive_ptr::decref(output_n);

  // Resize output
  if (is_batch == 0) {
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);
    THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
  }

  c10::raw::intrusive_ptr::decref(input);
  c10::raw::intrusive_ptr::decref(weight);
  if (bias) c10::raw::intrusive_ptr::decref(bias);
}

void THNN_(SpatialDilatedConvolution_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    THTensor *weight,
    THTensor *gradColumns,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH)
{
  THNN_(SpatialDilatedConvolution_shapeCheck)
    (input, gradOutput, weight, NULL, kH, kW, dH, dW, padH, padW,
     dilationH, dilationW, 0);

  // Params
  int64_t nInputPlane = weight->size(1);
  int64_t nOutputPlane = weight->size(0);

  input = THTensor_(newContiguous)(input);
  weight = THTensor_(newContiguous)(weight);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  THArgCheck(THTensor_(isContiguous)(gradColumns), 5, "gradColumns needs to be contiguous");
  int is_batch = 1;
  if (input->dim() == 3) {
    // Force batch
    is_batch = 0;
    THTensor_(resize4d)(input, 1, input->size(0), input->size(1), input->size(2));
    THTensor_(resize4d)(gradOutput, 1, gradOutput->size(0), gradOutput->size(1),
                        gradOutput->size(2));
  }

  int64_t inputWidth   = input->size(3);
  int64_t inputHeight  = input->size(2);
  int64_t outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  // Batch size + input planes
  int64_t batchSize = input->size(0);

  // Resize output
  THTensor_(resize4d)(gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  // Resize temporary columns
  THTensor_(resize2d)(gradColumns, nInputPlane*kW*kH, outputHeight*outputWidth);
  THTensor_(zero)(gradColumns);

  // Helpers
  THTensor *gradInput_n = THTensor_(new)();
  THTensor *gradOutput_n = THTensor_(new)();

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THTensor_(select)(gradInput_n, gradInput, 0, elt);
    THTensor_(select)(gradOutput_n, gradOutput, 0, elt);

    // M,N,K are dims of matrix A and B
    int64_t m = nInputPlane*kW*kH;
    int64_t n = gradColumns->size(1);
    int64_t k = nOutputPlane;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THBlas_(gemm)(
        'n', 't',
        n, m, k,
        1,
        gradOutput_n->data<scalar_t>(), n,
        weight->data<scalar_t>(), m,
        0,
        gradColumns->data<scalar_t>(), n
    );

    // Unpack columns back into input:
    THNN_(col2im)(
      gradColumns->data<scalar_t>(),
      nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
      kH, kW, padH, padW, dH, dW,
      dilationH, dilationW,
      gradInput_n->data<scalar_t>()
    );
  }

  // Free
  c10::raw::intrusive_ptr::decref(gradInput_n);
  c10::raw::intrusive_ptr::decref(gradOutput_n);

  // Resize output
  if (is_batch == 0) {
    THTensor_(resize3d)(gradOutput, nOutputPlane, outputHeight, outputWidth);
    THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
    THTensor_(resize3d)(gradInput, nInputPlane, inputHeight, inputWidth);
  }

  c10::raw::intrusive_ptr::decref(input);
  c10::raw::intrusive_ptr::decref(gradOutput);
  c10::raw::intrusive_ptr::decref(weight);
}


void THNN_(SpatialDilatedConvolution_accGradParameters)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradWeight,
    THTensor *gradBias,
    THTensor *columns,
    THTensor *ones,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH,
    accreal scale_)
{
  scalar_t scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  THNN_(SpatialDilatedConvolution_shapeCheck)
    (input, gradOutput, gradWeight, gradBias, kH, kW, dH, dW, padH, padW,
     dilationH, dilationW, 1);

  // Params
  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  if (gradWeight) {
    THArgCheck(THTensor_(isContiguous)(gradWeight), 4, "gradWeight needs to be contiguous");
  }
  THArgCheck(THTensor_(isContiguous)(columns), 6, "columns needs to be contiguous");
  if (gradBias) {
    THArgCheck(THTensor_(isContiguous)(gradBias), 5, "gradBias needs to be contiguous");
    THArgCheck(THTensor_(isContiguous)(ones), 7, "ones needs to be contiguous");
  }
  int is_batch = 1;
  if (input->dim() == 3) {
    // Force batch
    is_batch = 0;
    THTensor_(resize4d)(input, 1, input->size(0), input->size(1), input->size(2));
    THTensor_(resize4d)(gradOutput, 1, gradOutput->size(0),
                        gradOutput->size(1), gradOutput->size(2));
  }

  int64_t nInputPlane = input->size(1);
  int64_t nOutputPlane = gradOutput->size(1);
  int64_t inputWidth   = input->size(3);
  int64_t inputHeight  = input->size(2);
  int64_t outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  // Batch size + input planes
  int64_t batchSize = input->size(0);

  // Resize temporary columns
  THTensor_(resize2d)(columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Helpers
  THTensor *input_n = THTensor_(new)();
  THTensor *gradOutput_n = THTensor_(new)();

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THTensor_(select)(gradOutput_n, gradOutput, 0, elt);

    // Do Weight:
    if (gradWeight) {
      // Matrix mulitply per output:
      THTensor_(select)(input_n, input, 0, elt);

      // Extract columns:
      THNN_(im2col)(
        input_n->data<scalar_t>(),
        nInputPlane, inputHeight, inputWidth,
        outputHeight, outputWidth,
        kH, kW, padH, padW, dH, dW,
        dilationH, dilationW,
        columns->data<scalar_t>()
      );

      // M,N,K are dims of matrix A and B
      int64_t m = nOutputPlane;
      int64_t n = nInputPlane*kW*kH;
      int64_t k = columns->size(1);

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      THBlas_(gemm)(
          't', 'n',
          n, m, k,
          scale,
          columns->data<scalar_t>(), k,
          gradOutput_n->data<scalar_t>(), k,
          1,
          gradWeight->data<scalar_t>(), n
      );
    }

    // Do Bias:
    if (gradBias) {
      // M,N,K are dims of matrix A and B
      int64_t m_ = nOutputPlane;
      int64_t k_ = outputHeight * outputWidth;

      // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
      // Define a buffer of ones, for bias accumulation
      if (ones->dim() != 2 || ones->size(0)*ones->size(1) < outputHeight*outputWidth) {
        // Resize plane and fill with ones...
        THTensor_(resize2d)(ones, outputHeight, outputWidth);
        THTensor_(fill)(ones, 1);
      }
      THBlas_(gemv)(
          't',
          k_, m_,
          scale,
          gradOutput_n->data<scalar_t>(), k_,
          ones->data<scalar_t>(), 1,
          1,
          gradBias->data<scalar_t>(), 1
      );
    }
  }

  // Free
  c10::raw::intrusive_ptr::decref(input_n);
  c10::raw::intrusive_ptr::decref(gradOutput_n);

  // Resize
  if (is_batch == 0) {
    THTensor_(resize3d)(gradOutput, nOutputPlane, outputHeight, outputWidth);
    THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
  }

  c10::raw::intrusive_ptr::decref(input);
  c10::raw::intrusive_ptr::decref(gradOutput);
}

#endif
