#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialFullDilatedConvolution.cu"
#else

static inline void THNN_(SpatialFullDilatedConvolution_shapeCheck)(
                         THCState *state,
                         THCTensor *input, THCTensor *gradOutput,
                         THCTensor *weight, THCTensor *bias,
                         int kH, int kW, int dH, int dW, int padH, int padW,
                         int dilationH, int dilationW,
                         int adjH, int adjW) {
  THArgCheck(kW > 0 && kH > 0, 9,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THArgCheck(adjW < dW && adjH < dH, 15,
             "output adjustment must be smaller than stride, but got adjH: %d adjW: %d dH: %d dW: %d",
             adjH, adjW, dH, dW);
  THArgCheck(dilationW > 0 && dilationH > 0, 15,
             "dilation should be greater than zero, but got dilationH: %d, dilationW: %d",
             dilationH, dilationW);
  THArgCheck(THCTensor_(isContiguous)(state, weight), 4,
             "weight tensor has to be contiguous");
  THArgCheck(!bias || THCTensor_(isContiguous)(state, bias), 5,
             "bias tensor has to be contiguous");
  THCUNN_argCheck(state, weight->nDimension == 2 || weight->nDimension == 4, 5, weight,
                  "2D or 4D weight tensor expected, but got: %s");

  if (bias != NULL) {
    THCUNN_check_dim_size(state, bias, 1, 0, weight->size[1]);
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

  long nInputPlane  = weight->size[0];
  long inputHeight  = input->size[dimh];
  long inputWidth   = input->size[dimw];
  long nOutputPlane = weight->size[1];
  long outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1) + adjH;
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1) + adjW;

  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%d x %d x %d). "
            "Calculated output size: (%d x %d x %d). Output size is too small",
            nInputPlane,inputHeight,inputWidth,nOutputPlane,outputHeight,outputWidth);

  THCUNN_check_dim_size(state, input, ndim, dimf, nInputPlane);

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, ndim, dimf, nOutputPlane);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimh, outputHeight);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimw, outputWidth);
  }
}

void THNN_(SpatialFullDilatedConvolution_updateOutput)(
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
           int dilationW, int dilationH,
           int adjW, int adjH)
{

  int nInputPlane = THCTensor_(size)(state, weight, 0);
  int nOutputPlane = THCTensor_(size)(state, weight, 1);

  THCUNN_assertSameGPU(state, 6, input, output, weight,
                       bias, columns, ones);
  THNN_(SpatialFullDilatedConvolution_shapeCheck)
       (state, input, NULL, weight, bias, kH, kW, dH, dW, padH, padW, dilationH, dilationW, adjH, adjW);

  input = THCTensor_(newContiguous)(state, input);
  weight = THCTensor_(newContiguous)(state, weight);
  bias = bias ? THCTensor_(newContiguous)(state, bias) : bias;

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1) + adjH;
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1) + adjW;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCTensor_(resize4d)(state, output, batchSize, nOutputPlane, outputHeight, outputWidth);

  // Resize temporary columns
  THCTensor_(resize2d)(state, columns, nOutputPlane*kW*kH, inputHeight*inputWidth);

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

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[1] * weight->size[2] * weight->size[3];
    long n = columns->size[1];
    long k = weight->size[0];

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
        THCTensor_(data)(state, input_n), n,
        THCTensor_(data)(state, weight), m,
        ScalarConvert<int, real>::to(0),
        THCTensor_(data)(state, columns), n
    );

    // Unpack columns back into input:
    col2im<real, accreal>(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, columns),
      nOutputPlane, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
      dilationH, dilationW, THCTensor_(data)(state, output_n)
    );

    // Do Bias after:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long n_ = outputHeight * outputWidth;
    long k_ = 1;

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
          ScalarConvert<int, real>::to(1),
          THCTensor_(data)(state, output_n), n_
      );
    }
  }

  // Free
  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);

  // Resize output
  if (batch == 0) {
    THCTensor_(resize3d)(state, output, nOutputPlane, outputHeight, outputWidth);
    THCTensor_(resize3d)(state, input, nInputPlane, inputHeight, inputWidth);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, weight);
  if (bias) THCTensor_(free)(state, bias);
  
}

void THNN_(SpatialFullDilatedConvolution_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *weight,
           THCTensor *gradColumns,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int dilationW, int dilationH,
           int adjW, int adjH)
{
  int nInputPlane = THCTensor_(size)(state, weight, 0);
  int nOutputPlane = THCTensor_(size)(state, weight, 1);

  THCUNN_assertSameGPU(state, 5, input, gradOutput, weight,
                       gradColumns, gradInput);
  THNN_(SpatialFullDilatedConvolution_shapeCheck)
       (state, input, gradOutput, weight, NULL, kH, kW, dH, dW, padH, padW, dilationH, dilationW, adjH, adjW);

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  weight = THCTensor_(newContiguous)(state, weight);
  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THCTensor_(resize4d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1) + adjH;
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1) + adjW;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCTensor_(resize4d)(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  // Resize temporary columns
  THCTensor_(resize2d)(state, gradColumns, nOutputPlane*kW*kH, inputHeight*inputWidth);

  // Helpers
  THCTensor *gradInput_n = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THCTensor_(select)(state, gradInput_n, gradInput, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    im2col(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, gradOutput_n),
      nOutputPlane, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
      dilationH, dilationW, THCTensor_(data)(state, gradColumns)
    );


    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[0];
    long n = gradColumns->size[1];
    long k = weight->size[1] * weight->size[2] * weight->size[3];

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
        THCTensor_(data)(state, gradColumns), n,
        THCTensor_(data)(state, weight), k,
        ScalarConvert<int, real>::to(0),
        THCTensor_(data)(state, gradInput_n), n
    );
  }


  // Free
  THCTensor_(free)(state, gradInput_n);
  THCTensor_(free)(state, gradOutput_n);

  // Resize output
  if (batch == 0) {
    THCTensor_(resize3d)(state, gradOutput, nOutputPlane, outputHeight, outputWidth);
    THCTensor_(resize3d)(state, input, nInputPlane, inputHeight, inputWidth);
    THCTensor_(resize3d)(state, gradInput, nInputPlane, inputHeight, inputWidth);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
  THCTensor_(free)(state, weight);
}


void THNN_(SpatialFullDilatedConvolution_accGradParameters)(
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
           int adjW, int adjH,
           accreal scale_)
{
  real scale = ScalarConvert<accreal, real>::to(scale_);
  int nInputPlane = THCTensor_(size)(state, gradWeight, 0);
  int nOutputPlane = THCTensor_(size)(state, gradWeight, 1);

  THCUNN_assertSameGPU(state, 6, input, gradOutput, gradWeight,
                       gradBias, columns, ones);
  THNN_(SpatialFullDilatedConvolution_shapeCheck)
       (state, input, gradOutput, gradWeight, gradBias, kH, kW, dH, dW, padH, padW, dilationH, dilationW, adjH, adjW);

  THArgCheck(THCTensor_(isContiguous)(state, gradWeight), 4, "gradWeight needs to be contiguous");
  if (gradBias)
    THArgCheck(THCTensor_(isContiguous)(state, gradBias), 5, "gradBias needs to be contiguous");
  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THCTensor_(resize4d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1) + adjH;
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1) + adjW;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCTensor_(resize2d)(state, ones, outputHeight, outputWidth);
    THCTensor_(fill)(state, ones, ScalarConvert<int, real>::to(1));
  }

  // Resize temporary columns
  THCTensor_(resize2d)(state, columns, nOutputPlane*kW*kH, inputHeight*inputWidth);

  // Helpers
  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    im2col(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, gradOutput_n),
      nOutputPlane, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
      dilationH, dilationW, THCTensor_(data)(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long n = columns->size[0];   // nOutputPlane * kh * kw
    long m = input_n->size[0];   // nInputPlane
    long k = columns->size[1];   // inputHeight * inputWidth

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
        THCTensor_(data)(state, input_n), k,
        ScalarConvert<int, real>::to(1),
        THCTensor_(data)(state, gradWeight), n
    );

    // Do Bias:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long k_ = outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    if (gradBias) {
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
  if (batch == 0) {
    THCTensor_(resize3d)(state, gradOutput, nOutputPlane, outputHeight, outputWidth);
    THCTensor_(resize3d)(state, input, nInputPlane, inputHeight, inputWidth);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

#endif
