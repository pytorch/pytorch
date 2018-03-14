#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialFullDilatedConvolution.c"
#else

static inline void THNN_(SpatialFullDilatedConvolution_shapeCheck)(
	THTensor *input, THTensor *gradOutput,
	THTensor *weight, THTensor *bias,
	int kH, int kW, int dH, int dW, int padH, int padW,
	int dilationH, int dilationW, int adjH, int adjW, int weight_nullable) {

  THArgCheck(kW > 0 && kH > 0, 9,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 11,
	     "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THArgCheck(dilationW > 0 && dilationH > 0, 15,
             "dilation should be greater than zero, but got dilationH: %d, dilationW: %d",
             dilationH, dilationW);
  THArgCheck((adjW < dW || adjW < dilationW) && (adjH < dH || adjH < dilationH), 15,
             "output padding must be smaller than either stride or dilation, but got adjH: %d adjW: %d dH: %d dW: %d dilationH: %d dilationW: %d",
             adjH, adjW, dH, dW, dilationH, dilationW);

  if (weight != NULL) {
    THNN_ARGCHECK(weight->nDimension == 2 || weight->nDimension == 4, 5, weight,
  		"2D or 4D weight tensor expected, but got: %s");
    if (bias != NULL) {
      THNN_CHECK_DIM_SIZE(bias, 1, 0, weight->size[1]);
    }
  } else if (!weight_nullable) {
    THError("weight tensor is expected to be non-nullable");
  }

  int ndim = input->nDimension;
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THNN_ARGCHECK(ndim == 3 || ndim == 4, 2, input,
		"3D or 4D input tensor expected but got: %s");

  int64_t inputHeight  = input->size[dimh];
  int64_t inputWidth   = input->size[dimw];
  int64_t outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1) + adjH;
  int64_t outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1) + adjW;

  if (outputWidth < 1 || outputHeight < 1) {
    THError("Given input size per channel: (%ld x %ld). "
	    "Calculated output size per channel: (%ld x %ld). Output size is too small",
	    inputHeight, inputWidth, outputHeight, outputWidth);
  }

  if (weight != NULL) {
    int64_t nInputPlane = weight->size[0];
    THNN_CHECK_DIM_SIZE(input, ndim, dimf, nInputPlane);
  }

  if (gradOutput != NULL) {
    if (weight != NULL) {
      int64_t nOutputPlane = weight->size[1];
      THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    } else if (bias != NULL) {
      int64_t nOutputPlane = bias->size[0];
      THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    }
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
}

void THNN_(SpatialFullDilatedConvolution_updateOutput)(
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
    int dilationW, int dilationH,
    int adjW, int adjH)
{
  THNN_(SpatialFullDilatedConvolution_shapeCheck)
    (input, NULL, weight, bias, kH, kW, dH, dW, padH, padW,
     dilationH, dilationW, adjH, adjW, 0);

  int nInputPlane = THTensor_(size)(weight,0);
  int nOutputPlane = THTensor_(size)(weight,1);

  input = THTensor_(newContiguous)(input);
  weight = THTensor_(newContiguous)(weight);
  THArgCheck(THTensor_(isContiguous)(columns), 5, "columns needs to be contiguous");
  if (bias) {
    bias = THTensor_(newContiguous)(bias);
    THArgCheck(THTensor_(isContiguous)(ones), 6, "ones needs to be contiguous");
  }

  int is_batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    is_batch = 0;
    THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
  }

  int64_t inputHeight  = input->size[2];
  int64_t inputWidth   = input->size[3];
  int64_t outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1) + adjH;
  int64_t outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1) + adjW;

  // Batch size + input planes
  int64_t batchSize = input->size[0];

  // Resize output
  THTensor_(resize4d)(output, batchSize, nOutputPlane, outputHeight, outputWidth);

  // Resize temporary columns
  THTensor_(resize2d)(columns, nOutputPlane*kW*kH, inputHeight*inputWidth);
  THTensor_(zero)(columns);

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

  int elt;
  // For each elt in batch, do:
  for (elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(output_n, output, 0, elt);

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    int64_t m = weight->size[1] * weight->size[2] * weight->size[3];
    int64_t n = columns->size[1];
    int64_t k = weight->size[0];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THBlas_(gemm)(
        'n', 't',
        n, m, k,
        1,
        THTensor_(data)(input_n), n,
        THTensor_(data)(weight), m,
        0,
        THTensor_(data)(columns), n
    );

    // Unpack columns back into input:
    THNN_(col2im)(
      THTensor_(data)(columns),
      nOutputPlane, outputHeight, outputWidth, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      dilationH, dilationW,
      THTensor_(data)(output_n)
    );

    // Do Bias after:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    int64_t m_ = nOutputPlane;
    int64_t n_ = outputHeight * outputWidth;
    int64_t k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    if (bias) {
      THBlas_(gemm)(
          't', 'n',
          n_, m_, k_,
          1,
          THTensor_(data)(ones), k_,
          THTensor_(data)(bias), k_,
          1,
          THTensor_(data)(output_n), n_
      );
    }
  }

  // Free
  THTensor_(free)(input_n);
  THTensor_(free)(output_n);

  // Resize output
  if (is_batch == 0) {
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);
    THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
  }

  THTensor_(free)(input);
  THTensor_(free)(weight);
  if (bias) THTensor_(free)(bias);
}

void THNN_(SpatialFullDilatedConvolution_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    THTensor *weight,
    THTensor *gradColumns,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH,
    int adjW, int adjH)
{
  THNN_(SpatialFullDilatedConvolution_shapeCheck)
    (input, gradOutput, weight, NULL, kH, kW, dH, dW, padH, padW,
     dilationH, dilationW, adjH, adjW, 0);

  int nInputPlane = THTensor_(size)(weight,0);
  int nOutputPlane = THTensor_(size)(weight,1);

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  weight = THTensor_(newContiguous)(weight);
  THArgCheck(THTensor_(isContiguous)(gradColumns), 5, "gradColumns needs to be contiguous");

  int is_batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    is_batch = 0;
    THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
    THTensor_(resize4d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  int64_t inputWidth   = input->size[3];
  int64_t inputHeight  = input->size[2];
  int64_t outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1) + adjH;
  int64_t outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1) + adjW;

  // Batch size + input planes
  int64_t batchSize = input->size[0];

  // Resize output
  THTensor_(resize4d)(gradInput, batchSize, nInputPlane, inputHeight, inputWidth);
  THTensor_(zero)(gradInput);

  // Resize temporary columns
  THTensor_(resize2d)(gradColumns, nOutputPlane*kW*kH, inputHeight*inputWidth);

  // Helpers
  THTensor *gradInput_n = THTensor_(new)();
  THTensor *gradOutput_n = THTensor_(new)();

  int elt;
  // For each elt in batch, do:
  for (elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THTensor_(select)(gradInput_n, gradInput, 0, elt);
    THTensor_(select)(gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    THNN_(im2col)(
      THTensor_(data)(gradOutput_n),
      nOutputPlane, outputHeight, outputWidth,
      inputHeight, inputWidth,
      kH, kW, padH, padW, dH, dW,
      dilationH, dilationW,
      THTensor_(data)(gradColumns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    int64_t m = weight->size[0];
    int64_t n = gradColumns->size[1];
    int64_t k = weight->size[1] * weight->size[2] * weight->size[3];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THBlas_(gemm)(
        'n', 'n',
        n, m, k,
        1,
        THTensor_(data)(gradColumns), n,
        THTensor_(data)(weight), k,
        0,
        THTensor_(data)(gradInput_n), n
    );
  }

  // Free
  THTensor_(free)(gradInput_n);
  THTensor_(free)(gradOutput_n);

  // Resize output
  if (is_batch == 0) {
    THTensor_(resize3d)(gradOutput, nOutputPlane, outputHeight, outputWidth);
    THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
    THTensor_(resize3d)(gradInput, nInputPlane, inputHeight, inputWidth);
  }

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
  THTensor_(free)(weight);
}


void THNN_(SpatialFullDilatedConvolution_accGradParameters)(
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
    int adjW, int adjH,
    accreal scale_)
{
  real scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  THNN_(SpatialFullDilatedConvolution_shapeCheck)
    (input, gradOutput, gradWeight, gradBias, kH, kW, dH, dW, padH, padW,
     dilationH, dilationW, adjH, adjW, 1);

  int nOutputPlane;
  if (gradWeight) {
    nOutputPlane = THTensor_(size)(gradWeight, 1);
  } else if (gradBias) {
    nOutputPlane = THTensor_(size)(gradBias, 0);
  } else {
    return;
  }

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
  if (input->nDimension == 3) {
    // Force batch
    is_batch = 0;
    THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
    THTensor_(resize4d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  int64_t inputWidth   = input->size[3];
  int64_t inputHeight  = input->size[2];
  int64_t outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1) + adjH;
  int64_t outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1) + adjW;

  // Batch size + input planes
  int64_t batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THTensor_(resize2d)(ones, outputHeight, outputWidth);
    THTensor_(fill)(ones, 1);
  }

  // Resize temporary columns
  THTensor_(resize2d)(columns, nOutputPlane*kW*kH, inputHeight*inputWidth);

  // Helpers
  THTensor *input_n = THTensor_(new)();
  THTensor *gradOutput_n = THTensor_(new)();

  int elt;
  // For each elt in batch, do:
  for (elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THTensor_(select)(gradOutput_n, gradOutput, 0, elt);

    // Do Weight:
    if (gradWeight) {
      // Matrix mulitply per output:
      THTensor_(select)(input_n, input, 0, elt);

      // Extract columns:
      THNN_(im2col)(
        THTensor_(data)(gradOutput_n),
        nOutputPlane, outputHeight, outputWidth,
        inputHeight, inputWidth,
        kH, kW, padH, padW, dH, dW,
        dilationH, dilationW,
        THTensor_(data)(columns)
      );

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      int64_t n = columns->size[0];   // nOutputPlane * kh * kw
      int64_t m = input_n->size[0];   // nInputPlane
      int64_t k = columns->size[1];   // inputHeight * inputWidth

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      THBlas_(gemm)(
          't', 'n',
          n, m, k,
          scale,
          THTensor_(data)(columns), k,
          THTensor_(data)(input_n), k,
          1,
          THTensor_(data)(gradWeight), n
      );
    }

    // Do Bias:
    if (gradBias) {
      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      int64_t m_ = nOutputPlane;
      int64_t k_ = outputHeight * outputWidth;

      // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
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
  if (is_batch == 0) {
    THTensor_(resize3d)(gradOutput, nOutputPlane, outputHeight, outputWidth);
    THTensor_(resize3d)(input, input->size[1], inputHeight, inputWidth);
  }

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
}

#endif
