#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricDilatedConvolution.c"
#else

static inline void THNN_(VolumetricDilatedConvolution_shapeCheck)(
                         THTensor *input, THTensor *gradOutput,
                         THTensor *weight, THTensor *bias,
                         int kT, int kH, int kW, int dT, int dH, int dW,
                         int padT, int padH, int padW,
                         int dilationT, int dilationH, int dilationW) {
  THNN_ARGCHECK(input->nDimension == 4 || input->nDimension == 5, 2, input,
                "4D or 5D (batch mode) tensor expected for input, but got: %s");
  THNN_ARGCHECK(weight->nDimension == 5, 4, weight,
                "5D (nOutputPlane x nInputPlane x kT x kH x kW) tensor "
                "expected for weight, but got: %s");
  THArgCheck(kT > 0 && kW > 0 && kH > 0, 8,
             "kernel size should be greater than zero, but got kT: %d kH: %d kW: %d", kT, kH, kW);
  THArgCheck(dT > 0 && dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dT: %d dH: %d dW: %d", dT, dH, dW);
  THArgCheck(dilationT > 0 && dilationW > 0 && dilationH > 0, 15,
             "dilation should be greater than zero, but got dilationT: %d, dilationH: %d, dilationW: %d",
             dilationT, dilationH, dilationW);
  if (bias != NULL) {
    THNN_CHECK_DIM_SIZE(bias, 1, 0, weight->size[0]);
  }

  // Params
  int ndim = input->nDimension;
  int nInputPlane = weight->size[1];
  int nOutputPlane = weight->size[0];
  int dimf = 0;
  int dimd = 1;
  int dimh = 2;
  int dimw = 3;

  if (ndim == 5) {
    dimf++;
    dimd++;
    dimh++;
    dimw++;
  }

  long inputDepth  = input->size[dimd];
  long inputHeight  = input->size[dimh];
  long inputWidth   = input->size[dimw];
  long outputDepth  = (inputDepth  + 2*padT - (dilationT * (kT - 1) + 1)) / dT + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth  = (inputWidth  + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  if (outputDepth < 1 || outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%dx%dx%dx%d). Calculated output size: (%dx%dx%dx%d). Output size is too small",
            nInputPlane,inputDepth,inputHeight,inputWidth,nOutputPlane,outputDepth,outputHeight,outputWidth);

  THNN_CHECK_DIM_SIZE(input, ndim, dimf, nInputPlane);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimd, outputDepth);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
}

void THNN_(VolumetricDilatedConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *columns,
          THTensor *ones,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH)
{
  THNN_(VolumetricDilatedConvolution_shapeCheck)(
        input, NULL, weight, bias,
        kT, kH, kW, dT, dH, dW, padT, padH, padW,
        dilationT, dilationH, dilationW);

  // Params:
  int nInputPlane = weight->size[1];
  int nOutputPlane = weight->size[0];

  input = THTensor_(newContiguous)(input);
  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THTensor_(resize5d)(input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
  }

  long inputDepth  = input->size[2];
  long inputHeight  = input->size[3];
  long inputWidth   = input->size[4];
  long outputDepth  = (inputDepth  + 2*padT - (dilationT * (kT - 1) + 1)) / dT + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth  = (inputWidth  + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THTensor_(resize5d)(output, batchSize, nOutputPlane, outputDepth, outputHeight, outputWidth);
  THTensor_(zero)(output);

  // Resize temporary columns
  THTensor_(resize2d)(columns, nInputPlane*kT*kW*kH, outputDepth*outputHeight*outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 3 ||
      ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THTensor_(resize3d)(ones, outputDepth, outputHeight, outputWidth);
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
    long m_ = nOutputPlane;
    long n_ = outputDepth * outputHeight * outputWidth;
    long k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    if (bias) {
      THBlas_(gemm)(
        't', 'n',
        n_, m_, k_,
        1,
        THTensor_(data)(ones), k_,
        THTensor_(data)(bias), k_,
        0,
        THTensor_(data)(output_n), n_
      );
    } else {
      THTensor_(zero)(output_n);
    }

    // Extract columns:
    THNN_(vol2col)(
      THTensor_(data)(input_n),
      nInputPlane, inputDepth, inputHeight, inputWidth,
      kT, kH, kW, padT, padH, padW, dT, dH, dW,
      dilationT, dilationH, dilationW,
      THTensor_(data)(columns)
    );

    // M,N,K are dims of matrix A and B
    long m = nOutputPlane;
    long n = columns->size[1];
    long k = nInputPlane*kT*kH*kW;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THBlas_(gemm)(
      'n', 'n',
      n, m, k,
      1,
      THTensor_(data)(columns), n,
      THTensor_(data)(weight), k,
      1,
      THTensor_(data)(output_n), n
    );
  }

  // Free
  THTensor_(free)(input_n);
  THTensor_(free)(output_n);

  // Resize output
  if (batch == 0) {
    THTensor_(resize4d)(output, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THTensor_(resize4d)(input, nInputPlane, inputDepth, inputHeight, inputWidth);
  }

  THTensor_(free)(input);
}

void THNN_(VolumetricDilatedConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *gradColumns,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH)
{
  THNN_(VolumetricDilatedConvolution_shapeCheck)(
        input, gradOutput, weight, NULL,
        kT, kH, kW, dT, dH, dW, padT, padH, padW,
        dilationT, dilationH, dilationW);

  // Params
  int nInputPlane = weight->size[1];
  int nOutputPlane = weight->size[0];

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THTensor_(resize5d)(input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THTensor_(resize5d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputDepth  = input->size[2];
  long inputWidth   = input->size[4];
  long inputHeight  = input->size[3];
  long outputDepth  = (inputDepth + 2*padT - (dilationT * (kT - 1) + 1)) / dT + 1;
  long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THTensor_(resize5d)(gradInput, batchSize, nInputPlane, inputDepth, inputHeight, inputWidth);

  // Resize temporary columns
  THTensor_(resize2d)(gradColumns, nInputPlane*kT*kW*kH, outputDepth*outputHeight*outputWidth);
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
    long m = nInputPlane*kT*kW*kH;
    long n = gradColumns->size[1];
    long k = nOutputPlane;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THBlas_(gemm)(
        'n', 't',
        n, m, k,
        1,
        THTensor_(data)(gradOutput_n), n,
        THTensor_(data)(weight), m,
        0,
        THTensor_(data)(gradColumns), n
    );

    // Unpack columns back into input:
    THNN_(col2vol)(
      THTensor_(data)(gradColumns),
      nInputPlane, inputDepth, inputHeight, inputWidth,
      kT, kH, kW, padT, padH, padW, dT, dH, dW,
      dilationT, dilationH, dilationW,
      THTensor_(data)(gradInput_n)
    );
  }

  // Free
  THTensor_(free)(gradInput_n);
  THTensor_(free)(gradOutput_n);

  // Resize output
  if (batch == 0) {
    THTensor_(resize4d)(gradOutput, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THTensor_(resize4d)(input, nInputPlane, inputDepth, inputHeight, inputWidth);
    THTensor_(resize4d)(gradInput, nInputPlane, inputDepth, inputHeight, inputWidth);
  }

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
}

void THNN_(VolumetricDilatedConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *columns,
          THTensor *ones,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH,
          real scale)
{
  THNN_(VolumetricDilatedConvolution_shapeCheck)(
        input, gradOutput, gradWeight, gradBias,
        kT, kH, kW, dT, dH, dW, padT, padH, padW,
        dilationT, dilationH, dilationW);

  // Params
  int nInputPlane = gradWeight->size[1];
  int nOutputPlane = gradWeight->size[0];

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THTensor_(resize5d)(input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THTensor_(resize5d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputDepth  = input->size[2];
  long inputWidth   = input->size[4];
  long inputHeight  = input->size[3];
  long outputDepth  = (inputDepth + 2*padT - (dilationT * (kT - 1) + 1)) / dT + 1;
  long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 3 || ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THTensor_(resize3d)(ones, outputDepth, outputHeight, outputWidth);
    THTensor_(fill)(ones, 1);
  }

  // Resize temporary columns
  THTensor_(resize2d)(columns, nInputPlane*kT*kW*kH, outputDepth*outputHeight*outputWidth);

  // Helpers
  THTensor *input_n = THTensor_(new)();
  THTensor *gradOutput_n = THTensor_(new)();

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    THNN_(vol2col)(
      THTensor_(data)(input_n),
      nInputPlane, inputDepth, inputHeight, inputWidth,
      kT, kH, kW, padT, padH, padW, dT, dH, dW,
      dilationT, dilationH, dilationW,
      THTensor_(data)(columns)
    );

    // M,N,K are dims of matrix A and B
    long m = nOutputPlane;
    long n = nInputPlane*kT*kW*kH;
    long k = columns->size[1];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THBlas_(gemm)(
        't', 'n',
        n, m, k,
        scale,
        THTensor_(data)(columns), k,
        THTensor_(data)(gradOutput_n), k,
        1,
        THTensor_(data)(gradWeight), n
    );

    // Do Bias:
    // M,N,K are dims of matrix A and B
    long m_ = nOutputPlane;
    long k_ = outputDepth * outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    if (gradBias) {
      THBlas_(gemv)(
          't',
          k_, m_,
          scale,
          THTensor_(data)(gradOutput_n), k_,
          THTensor_(data)(ones), 1,
          1,
          THTensor_(data)(gradBias), 1
      );
    }
  }

  // Free
  THTensor_(free)(input_n);
  THTensor_(free)(gradOutput_n);

  // Resize
  if (batch == 0) {
    THTensor_(resize4d)(gradOutput, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THTensor_(resize4d)(input, nInputPlane, inputDepth, inputHeight, inputWidth);
  }

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
}

#endif
