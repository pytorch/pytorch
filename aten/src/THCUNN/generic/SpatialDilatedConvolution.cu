#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialDilatedConvolution.cu"
#else

static inline void THNN_(SpatialDilatedConvolution_shapeCheck)(
                         THCState *state,
                         THCTensor *input, THCTensor *gradOutput,
                         THCTensor *weight, THCTensor *bias,
                         int kH, int kW, int dH, int dW, int padH, int padW,
                         int dilationH, int dilationW, int weight_nullable) {
  THArgCheck(kW > 0 && kH > 0, 9,
	           "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THArgCheck(dilationW > 0 && dilationH > 0, 14,
             "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
             dilationH, dilationW);

  if (weight != NULL) {
    THCUNN_argCheck(state, weight->nDimension == 4, 4, weight,
                    "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
                  "but got: %s");
    if (bias != NULL) {
      THCUNN_check_dim_size(state, bias, 1, 0, weight->size[0]);
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

   THCUNN_argCheck(state, ndim == 3 || ndim == 4, 2, input,
                   "3D or 4D input tensor expected but got: %s");

   int64_t inputHeight  = input->size[dimh];
   int64_t inputWidth   = input->size[dimw];

   int64_t outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;
   int64_t outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;

   if (outputWidth < 1 || outputHeight < 1) {
    THError("Given input size per channel: (%ld x %ld). "
      "Calculated output size per channel: (%ld x %ld). Output size is too small",
      inputHeight, inputWidth, outputHeight, outputWidth);
  }

  if (weight != NULL) {
    int64_t nInputPlane = weight->size[1];
     THCUNN_check_dim_size(state, input, ndim, dimf, nInputPlane);
  }

   if (gradOutput != NULL) {
    if (weight != NULL) {
      int64_t nOutputPlane = weight->size[0];
      THCUNN_check_dim_size(state, gradOutput, ndim, dimf, nOutputPlane);
    } else if (bias != NULL) {
      int64_t nOutputPlane = bias->size[0];
      THCUNN_check_dim_size(state, gradOutput, ndim, dimf, nOutputPlane);
    }
     THCUNN_check_dim_size(state, gradOutput, ndim, dimh, outputHeight);
     THCUNN_check_dim_size(state, gradOutput, ndim, dimw, outputWidth);
   }
}

void THNN_(SpatialDilatedConvolution_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias,
           THCTensor *columns,
           THCTensor *ones,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int dilationW, int dilationH) {

  THCUNN_assertSameGPU(state, 5, input, output, weight, columns, ones);
  if (bias) {
    THCUNN_assertSameGPU(state, 2, weight, bias);
    THArgCheck(THCTensor_(isContiguous)(state, bias), 5, "bias tensor has to be contiguous");
  }
  THNN_(SpatialDilatedConvolution_shapeCheck)
       (state, input, NULL, weight, bias, kH, kW, dH, dW, padH, padW,
        dilationH, dilationW, 0);

  // Params:
  int nInputPlane = weight->size[1];
  int nOutputPlane = weight->size[0];

  input = THCTensor_(newContiguous)(state, input);
  weight = THCTensor_(newContiguous)(state, weight);
  bias = bias ? THCTensor_(newContiguous)(state, bias) : bias;

  int is_batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    is_batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
  }

  int64_t inputWidth   = input->size[3];
  int64_t inputHeight  = input->size[2];
  int64_t outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  // Batch size + input planes
  int64_t batchSize = input->size[0];

  // Resize output
  THCTensor_(resize4d)(state, output, batchSize, nOutputPlane, outputHeight, outputWidth);

  // Resize temporary columns
  THCTensor_(resize2d)(state, columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCTensor_(resize2d)(state, ones, outputHeight, outputWidth);
    THCTensor_(fill)(state, ones, ScalarConvert<int, real>::to(1));
  }

  // Helpers
  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *output_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    int64_t m_ = nOutputPlane;
    int64_t n_ = outputHeight * outputWidth;
    int64_t k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    if (bias) {
      #ifdef THC_REAL_IS_FLOAT
      THCudaBlas_Sgemm(
      #elif defined(THC_REAL_IS_HALF)
      THCudaBlas_Hgemm(
      #elif defined(THC_REAL_IS_DOUBLE)
      THCudaBlas_Dgemm(
      #endif
          state,
          't', 'n',
          n_, m_, k_,
          ScalarConvert<int, real>::to(1),
          THCTensor_(data)(state, ones), k_,
          THCTensor_(data)(state, bias), k_,
          ScalarConvert<int, real>::to(0),
          THCTensor_(data)(state, output_n), n_
      );
    } else {
      THCTensor_(zero)(state, output_n);
    }

    // Extract columns:
    im2col(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input_n),
      nInputPlane, inputHeight, inputWidth,
      outputHeight, outputWidth,
      kH, kW, padH, padW, dH, dW,
      dilationH, dilationW,
      THCTensor_(data)(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    int64_t m = nOutputPlane;
    int64_t n = columns->size[1];
    int64_t k = nInputPlane*kH*kW;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    #ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sgemm(
    #elif defined(THC_REAL_IS_HALF)
    THCudaBlas_Hgemm(
    #elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dgemm(
    #endif
        state,
        'n', 'n',
        n, m, k,
        ScalarConvert<int, real>::to(1),
        THCTensor_(data)(state, columns), n,
        THCTensor_(data)(state, weight), k,
        ScalarConvert<int, real>::to(1),
        THCTensor_(data)(state, output_n), n
    );
  }

  // Free
  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);

  // Resize output
  if (is_batch == 0) {
    THCTensor_(resize3d)(state, output, nOutputPlane, outputHeight, outputWidth);
    THCTensor_(resize3d)(state, input, nInputPlane, inputHeight, inputWidth);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, weight);
  if (bias) THCTensor_(free)(state, bias);
}

void THNN_(SpatialDilatedConvolution_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *weight,
           THCTensor *gradColumns,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int dilationW, int dilationH) {

  THCUNN_assertSameGPU(state, 5, input, gradOutput, weight,
                       gradColumns, gradInput);
  THNN_(SpatialDilatedConvolution_shapeCheck)
       (state, input, gradOutput, weight, NULL, kH, kW, dH, dW, padH, padW,
        dilationH, dilationW, 0);

  // Params
  int nInputPlane = weight->size[1];
  int nOutputPlane = weight->size[0];

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  weight = THCTensor_(newContiguous)(state, weight);

  int is_batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    is_batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THCTensor_(resize4d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  int64_t inputWidth   = input->size[3];
  int64_t inputHeight  = input->size[2];
  int64_t outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  // Batch size + input planes
  int64_t batchSize = input->size[0];

  // Resize output
  THCTensor_(resize4d)(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  // Resize temporary columns
  THCTensor_(resize2d)(state, gradColumns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Helpers
  THCTensor *gradInput_n = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THCTensor_(select)(state, gradInput_n, gradInput, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    int64_t m = nInputPlane*kW*kH;
    int64_t n = gradColumns->size[1];
    int64_t k = nOutputPlane;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    #ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sgemm(
    #elif defined(THC_REAL_IS_HALF)
    THCudaBlas_Hgemm(
    #elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dgemm(
    #endif
        state,
        'n', 't',
        n, m, k,
        ScalarConvert<int, real>::to(1),
        THCTensor_(data)(state, gradOutput_n), n,
        THCTensor_(data)(state, weight), m,
        ScalarConvert<int, real>::to(0),
        THCTensor_(data)(state, gradColumns), n
    );

    // Unpack columns back into input:
    col2im<real, accreal>(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, gradColumns),
      nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
      dilationH, dilationW,
      THCTensor_(data)(state, gradInput_n)
    );
  }

  // Free
  THCTensor_(free)(state, gradInput_n);
  THCTensor_(free)(state, gradOutput_n);

  // Resize output
  if (is_batch == 0) {
    THCTensor_(resize3d)(state, gradOutput, nOutputPlane, outputHeight, outputWidth);
    THCTensor_(resize3d)(state, input, nInputPlane, inputHeight, inputWidth);
    THCTensor_(resize3d)(state, gradInput, nInputPlane, inputHeight, inputWidth);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
  THCTensor_(free)(state, weight);
}

void THNN_(SpatialDilatedConvolution_accGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *columns,
           THCTensor *ones,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int dilationW, int dilationH,
           accreal scale_) {

  real scale = ScalarConvert<accreal, real>::to(scale_);
  THCUNN_assertSameGPU(state, 5, input, gradOutput, gradWeight, columns, ones);
  if (gradBias) {
   THCUNN_assertSameGPU(state, 2, gradWeight, gradBias);
  }
  THNN_(SpatialDilatedConvolution_shapeCheck)
       (state, input, gradOutput, gradWeight, gradBias, kH, kW, dH, dW, padH, padW,
        dilationH, dilationW, 1);

  if (gradWeight) {
    THArgCheck(THCTensor_(isContiguous)(state, gradWeight), 4, "gradWeight needs to be contiguous");
  }
  if (gradBias) {
    THArgCheck(THCTensor_(isContiguous)(state, gradBias), 5, "gradBias needs to be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, ones), 7, "ones needs to be contiguous");
  }

  // Params
  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  int is_batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    is_batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THCTensor_(resize4d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  int64_t nInputPlane = input->size[1];
  int64_t nOutputPlane = gradOutput->size[1];
  int64_t inputWidth   = input->size[3];
  int64_t inputHeight  = input->size[2];
  int64_t outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int64_t outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  // Batch size + input planes
  int64_t batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCTensor_(resize2d)(state, ones, outputHeight, outputWidth);
    THCTensor_(fill)(state, ones, ScalarConvert<int, real>::to(1));
  }

  // Resize temporary columns
  THCTensor_(resize2d)(state, columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Helpers
  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    // Do Weight:
    if (gradWeight) {
      // Matrix mulitply per output:
      THCTensor_(select)(state, input_n, input, 0, elt);

      // Extract columns:
      im2col(
        THCState_getCurrentStream(state),
        THCTensor_(data)(state, input_n),
        nInputPlane, inputHeight, inputWidth,
        outputHeight, outputWidth,
        kH, kW, padH, padW, dH, dW,
        dilationH, dilationW,
        THCTensor_(data)(state, columns)
      );

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      int64_t m = nOutputPlane;
      int64_t n = nInputPlane*kW*kH;
      int64_t k = columns->size[1];

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      #ifdef THC_REAL_IS_FLOAT
      THCudaBlas_Sgemm(
      #elif defined(THC_REAL_IS_HALF)
      THCudaBlas_Hgemm(
      #elif defined(THC_REAL_IS_DOUBLE)
      THCudaBlas_Dgemm(
      #endif
          state,
          't', 'n',
          n, m, k,
          scale,
          THCTensor_(data)(state, columns), k,
          THCTensor_(data)(state, gradOutput_n), k,
          ScalarConvert<int, real>::to(1),
          THCTensor_(data)(state, gradWeight), n
      );
    }

    // Do Bias:
    if (gradBias) {
      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      int64_t m_ = nOutputPlane;
      int64_t k_ = outputHeight * outputWidth;

      // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
      #if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
      #ifdef THC_REAL_IS_FLOAT
      THCudaBlas_Sgemv(
      #elif defined(THC_REAL_IS_DOUBLE)
      THCudaBlas_Dgemv(
      #endif
          state,
          't',
          k_, m_,
          scale,
          THCTensor_(data)(state, gradOutput_n), k_,
          THCTensor_(data)(state, ones), 1,
          ScalarConvert<int, real>::to(1),
          THCTensor_(data)(state, gradBias), 1
      );
      #endif
      #ifdef THC_REAL_IS_HALF
      THCudaBlas_Hgemm(
          state,
          't', 'n',
          m_, 1, k_,
          scale,
          THCTensor_(data)(state, gradOutput_n), k_,
          THCTensor_(data)(state, ones), k_,
          ScalarConvert<int, real>::to(1),
          THCTensor_(data)(state, gradBias), m_
      );
      #endif
    }
  }

  // Free
  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, gradOutput_n);

  // Resize
  if (is_batch == 0) {
    THCTensor_(resize3d)(state, gradOutput, nOutputPlane, outputHeight, outputWidth);
    THCTensor_(resize3d)(state, input, nInputPlane, inputHeight, inputWidth);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

#endif
