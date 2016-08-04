#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialDilatedConvolution.c"
#else

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
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");
  THArgCheck(weight->nDimension == 4, 4, "weight tensor must be 4D (nOutputPlane,nInputPlane,kH,kW)");
  THArgCheck(!bias || weight->size[0] == bias->size[0], 4, "nOutputPlane mismatch in weight and bias");
  THArgCheck(kW > 0 && kH > 0, 8, "kernel size should be greater than zero");
  THArgCheck(dW > 0 && dH > 0, 10, "stride should be greater than zero");

  // Params:
  int nInputPlane = weight->size[1];
  int nOutputPlane = weight->size[0];

  int batch = 1;
  if (input->nDimension == 3) {
    THArgCheck(input->size[0] == nInputPlane, 2, "input channels and nInputPlane dont match");
    // Force batch
    batch = 0;
    THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
  } else {
    THArgCheck(input->size[1] == nInputPlane, 2, "input channels and nInputPlane dont match");
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
            nInputPlane,inputHeight,inputWidth,nOutputPlane,outputHeight,outputWidth);

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THTensor_(resize4d)(output, batchSize, nOutputPlane, outputHeight, outputWidth);
  THTensor_(zero)(output);

  // Resize temporary columns
  THTensor_(resize2d)(columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
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
    long m_ = nOutputPlane;
    long n_ = outputHeight * outputWidth;
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
    THNN_(im2col)(
      THTensor_(data)(input_n),
      nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      dilationH, dilationW,
      THTensor_(data)(columns)
    );

    // M,N,K are dims of matrix A and B
    long m = nOutputPlane;
    long n = columns->size[1];
    long k = nInputPlane*kH*kW;

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
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);
    THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
  }
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
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");
  THArgCheck(weight->nDimension == 4, 4, "weight tensor must be 4D (nOutputPlane,nInputPlane,kH,kW)");
  THArgCheck(kW > 0 && kH > 0, 9, "kernel size should be greater than zero");
  THArgCheck(dW > 0 && dH > 0, 11, "stride should be greater than zero");

  // Params
  int nInputPlane = weight->size[1];
  int nOutputPlane = weight->size[0];

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
    THTensor_(resize4d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

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
    long m = nInputPlane*kW*kH;
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
    THNN_(col2im)(
      THTensor_(data)(gradColumns),
      nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      dilationH, dilationW,
      THTensor_(data)(gradInput_n)
    );
  }

  // Free
  THTensor_(free)(gradInput_n);
  THTensor_(free)(gradOutput_n);

  // Resize output
  if (batch == 0) {
    THTensor_(resize3d)(gradOutput, nOutputPlane, outputHeight, outputWidth);
    THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
    THTensor_(resize3d)(gradInput, nInputPlane, inputHeight, inputWidth);
  }
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
    real scale)
{
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");
  THArgCheck(gradWeight->nDimension == 4, 4, "gradWeight tensor must be 4D (nOutputPlane,nInputPlane,kH,kW)");
  THArgCheck(!gradBias || gradWeight->size[0] == gradBias->size[0], 4, "nOutputPlane mismatch in gradWeight and gradBias");
  THArgCheck(kW > 0 && kH > 0, 8, "kernel size should be greater than zero");
  THArgCheck(dW > 0 && dH > 0, 10, "stride should be greater than zero");

  // Params
  int nInputPlane = gradWeight->size[1];
  int nOutputPlane = gradWeight->size[0];

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
    THTensor_(resize4d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THTensor_(resize2d)(ones, outputHeight, outputWidth);
    THTensor_(fill)(ones, 1);
  }

  // Resize temporary columns
  THTensor_(resize2d)(columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Helpers
  THTensor *input_n = THTensor_(new)();
  THTensor *gradOutput_n = THTensor_(new)();

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    THNN_(im2col)(
      THTensor_(data)(input_n),
      nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      dilationH, dilationW,
      THTensor_(data)(columns)
    );

    // M,N,K are dims of matrix A and B
    long m = nOutputPlane;
    long n = nInputPlane*kW*kH;
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
    long k_ = outputHeight * outputWidth;

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
    THTensor_(resize3d)(gradOutput, nOutputPlane, outputHeight, outputWidth);
    THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
  }
}

#endif
