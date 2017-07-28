#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricFullDilatedConvolution.cu"
#else

static inline void THNN_(VolumetricFullDilatedConvolution_shapeCheck)(
               THCState *state,
               THCTensor *input,
               THCTensor *gradOutput,
               THCTensor *weight,
               THCTensor *bias,
               int dT, int dW, int dH,
               int padT, int padW, int padH,
               int dilationT, int dilationW, int dilationH,
               int adjT, int adjW, int adjH) {
  THCUNN_argCheck(state, input->nDimension == 4 || input->nDimension == 5, 2, input,
            "4D or 5D (batch mode) tensor expected for input, but got: %s");
   // number of input & output planes and kernel size is indirectly defined by the weight tensor
  THCUNN_argCheck(state, weight->nDimension == 5, 4, weight,
            "5D (nOutputPlane x nInputPlane x kT x kH x kW) tensor "
            "expected for weight, but got: %s");
  THArgCheck(THCTensor_(isContiguous)(state, weight), 4,
         "weight tensor has to be contiguous");
  THArgCheck(!bias || THCTensor_(isContiguous)(state, bias), 5,
         "bias tensor has to be contiguous");
  THArgCheck(dT > 0 && dW > 0 && dH > 0, 8,
         "stride should be greater than zero, but got dT: %d dH: %d dW: %d", dT, dH, dW);
  THArgCheck(adjT < dT && adjW < dW && adjH < dH, 14,
         "output adjustment must be smaller than stride, but got "
         "adjT: %d adjH: %d adjW: %d dT: %d dH: %d dW: %d",
         adjT, adjH, adjW, dT, dH, dW);
  THArgCheck(dilationT > 0 && dilationW > 0 && dilationH > 0, 15,
             "dilation should be greater than zero, but got dilationT: %d, dilationH: %d, dilationW: %d",
             dilationT, dilationH, dilationW);

  int ndim = input->nDimension;
  int nInputPlane = THCTensor_(size)(state, weight, 0);
  int nOutputPlane = THCTensor_(size)(state, weight, 1);
  const int kT       = (int)weight->size[2];
  const int kH       = (int)weight->size[3];
  const int kW       = (int)weight->size[4];

  if (bias != NULL) {
    THCUNN_check_dim_size(state, bias, 1, 0, weight->size[1]);
  }

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

  long inputWidth   = input->size[dimw];
  long inputHeight  = input->size[dimh];
  long inputDepth  = input->size[dimd];
  long outputDepth  = (inputDepth - 1) * dT - 2*padT + (dilationT * (kT - 1) + 1) + adjT;
  long outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1) + adjH;
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1) + adjW;
  if (outputDepth < 1 || outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%dx%dx%dx%d). Calculated output size: (%dx%dx%dx%d). Output size is too small",
        nInputPlane,inputDepth,inputHeight,inputWidth,nOutputPlane,outputDepth,outputHeight,outputWidth);

  THCUNN_check_dim_size(state, input, ndim, dimf, nInputPlane);
  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, ndim, dimf, nOutputPlane);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimd, outputDepth);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimh, outputHeight);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimw, outputWidth);
  }
}

void THNN_(VolumetricFullDilatedConvolution_updateOutput)(
       THCState *state,
       THCTensor  *input,
       THCTensor  *output,
       THCTensor  *weight,
       THCTensor  *bias,
       THCTensor  *finput,
       THCTensor  *fgradInput,
       int dT, int dW, int dH,
       int padT, int padW, int padH,
       int dilationT, int dilationW, int dilationH,
       int adjT, int adjW, int adjH)
{

  THCTensor  *columns = finput;
  THCTensor  *ones    = fgradInput;

  int nInputPlane = THCTensor_(size)(state, weight, 0);
  int nOutputPlane = THCTensor_(size)(state, weight, 1);
  const int kT       = (int)weight->size[2];
  const int kH       = (int)weight->size[3];
  const int kW       = (int)weight->size[4];

  THCUNN_assertSameGPU(state, 6, input, output, weight,
               bias, columns, ones);
  THNN_(VolumetricFullDilatedConvolution_shapeCheck)(
      state, input, NULL, weight, bias,
      dT, dW, dH, padT, padW, padH, dilationT, dilationW, dilationH,
      adjT, adjW, adjH);

  input = THCTensor_(newContiguous)(state, input);
  weight = THCTensor_(newContiguous)(state, weight);
  bias = bias ? THCTensor_(newContiguous)(state, bias) : bias;

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THCTensor_(resize5d)(state, input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
  }

  long inputWidth   = input->size[4];
  long inputHeight  = input->size[3];
  long inputDepth  = input->size[2];
  long outputDepth  = (inputDepth - 1) * dT - 2*padT + (dilationT * (kT - 1) + 1) + adjT;
  long outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1) + adjH;
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1) + adjW;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCTensor_(resize5d)(state, output, batchSize, nOutputPlane, outputDepth, outputHeight, outputWidth);

  // Resize temporary columns
  THCTensor_(resize2d)(state, columns, nOutputPlane*kW*kH*kT, inputDepth*inputHeight*inputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 3 || ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCTensor_(resize3d)(state, ones, outputDepth, outputHeight, outputWidth);
    THCTensor_(fill)(state, ones, ScalarConvert<int, real>::to(1));
  }

  // Helpers
  THCTensor  *input_n = THCTensor_(new)(state);
  THCTensor  *output_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[1] * weight->size[2] * weight->size[3] * weight->size[4];
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
    col2vol<real, accreal>(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, columns),
      nOutputPlane, outputDepth, outputHeight, outputWidth, kT, kH, kW, padT, padH, padW, dT, dH, dW,
      dilationT, dilationH, dilationW,
      THCTensor_(data)(state, output_n)
    );

    // Do Bias after:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long n_ = outputDepth * outputHeight * outputWidth;
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
    THCTensor_(resize4d)(state, output, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THCTensor_(resize4d)(state, input, nInputPlane, inputDepth, inputHeight, inputWidth);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, weight);
  if (bias) THCTensor_(free)(state, bias);

}

void THNN_(VolumetricFullDilatedConvolution_updateGradInput)(
       THCState *state,
       THCTensor  *input,
       THCTensor  *gradOutput,
       THCTensor  *gradInput,
       THCTensor  *weight,
       THCTensor  *finput,
       THCTensor  *fgradInput,
       int dT, int dW, int dH,
       int padT, int padW, int padH,
       int dilationT, int dilationW, int dilationH,
       int adjT, int adjW, int adjH)
{
  THCTensor  *gradColumns = finput;

  int nInputPlane = THCTensor_(size)(state, weight, 0);
  int nOutputPlane = THCTensor_(size)(state, weight, 1);
  const int kT       = (int)weight->size[2];
  const int kH       = (int)weight->size[3];
  const int kW       = (int)weight->size[4];

  THCUNN_assertSameGPU(state, 5, input, gradOutput, weight,
               gradColumns, gradInput);
  THNN_(VolumetricFullDilatedConvolution_shapeCheck)(
      state, input, gradOutput, weight, NULL,
      dT, dW, dH, padT, padW, padH, dilationT, dilationW, dilationH,
      adjT, adjW, adjH);

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  weight = THCTensor_(newContiguous)(state, weight);
  
  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THCTensor_(resize5d)(state, input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THCTensor_(resize5d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputWidth   = input->size[4];
  long inputHeight  = input->size[3];
  long inputDepth   = input->size[2];
  long outputDepth  = (inputDepth - 1) * dT - 2*padT + (dilationT * (kT - 1) + 1) + adjT;
  long outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1) + adjH;
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1) + adjW;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCTensor_(resize5d)(state, gradInput, batchSize, nInputPlane, inputDepth, inputHeight, inputWidth);

  // Resize temporary columns
  THCTensor_(resize2d)(state, gradColumns, nOutputPlane*kW*kH*kT, inputDepth*inputHeight*inputWidth);

  // Helpers
  THCTensor  *gradInput_n = THCTensor_(new)(state);
  THCTensor  *gradOutput_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THCTensor_(select)(state, gradInput_n, gradInput, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    vol2col(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, gradOutput_n),
      nOutputPlane, outputDepth, outputHeight, outputWidth, kT, kH, kW, padT, padH, padW, dT, dH, dW,
      dilationT, dilationH, dilationW,
      THCTensor_(data)(state, gradColumns)
    );


    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[0];
    long n = gradColumns->size[1];
    long k = weight->size[1] * weight->size[2] * weight->size[3] * weight->size[4];

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
    THCTensor_(resize4d)(state, gradOutput, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THCTensor_(resize4d)(state, input, nInputPlane, inputDepth, inputHeight, inputWidth);
    THCTensor_(resize4d)(state, gradInput, nInputPlane, inputDepth, inputHeight, inputWidth);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
  THCTensor_(free)(state, weight);
}


void THNN_(VolumetricFullDilatedConvolution_accGradParameters)(
           THCState *state,
           THCTensor  *input,
           THCTensor  *gradOutput,
           THCTensor  *gradWeight,
           THCTensor  *gradBias,
           THCTensor  *finput,
           THCTensor  *fgradInput,
           int dT, int dW, int dH,
           int padT, int padW, int padH,
           int dilationT, int dilationW, int dilationH,
           int adjT, int adjW, int adjH,
           accreal scale_)
{
  real scale = ScalarConvert<accreal, real>::to(scale_);
  THCTensor  *columns = finput;
  THCTensor  *ones = fgradInput;

  int nInputPlane = THCTensor_(size)(state, gradWeight, 0);
  int nOutputPlane = THCTensor_(size)(state, gradWeight, 1);
  const int kT       = (int)gradWeight->size[2];
  const int kH       = (int)gradWeight->size[3];
  const int kW       = (int)gradWeight->size[4];

  THCUNN_assertSameGPU(state, 6, input, gradOutput, gradWeight,
               gradBias, columns, ones);
  THNN_(VolumetricFullDilatedConvolution_shapeCheck)(
      state, input, gradOutput, gradWeight,
      gradBias, dT, dW, dH, padT, padW, padH, dilationT, dilationW, dilationH,
      adjT, adjW, adjH);

  THArgCheck(THCTensor_(isContiguous)(state, gradWeight), 4, "gradWeight needs to be contiguous");
  if (gradBias)
    THArgCheck(THCTensor_(isContiguous)(state, gradBias), 5, "gradBias needs to be contiguous");

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THCTensor_(resize5d)(state, input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THCTensor_(resize5d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputWidth   = input->size[4];
  long inputHeight  = input->size[3];
  long inputDepth   = input->size[2];
  long outputDepth  = (inputDepth - 1) * dT - 2*padT + (dilationT * (kT - 1) + 1) + adjT;
  long outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1) + adjH;
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1) + adjW;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 3 || ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCTensor_(resize3d)(state, ones, outputDepth, outputHeight, outputWidth);
    THCTensor_(fill)(state, ones, ScalarConvert<int, real>::to(1));
  }

  // Resize temporary columns
  THCTensor_(resize2d)(state, columns, nOutputPlane*kW*kH*kT, inputDepth*inputHeight*inputWidth);

  // Helpers
  THCTensor  *input_n = THCTensor_(new)(state);
  THCTensor  *gradOutput_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    vol2col(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, gradOutput_n),
      nOutputPlane, outputDepth, outputHeight, outputWidth, kT, kH, kW, padT, padH, padW, dT, dH, dW,
      dilationT, dilationH, dilationW,
      THCTensor_(data)(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long n = columns->size[0];   // nOutputPlane * kt * kh * kw
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
    long k_ = outputDepth * outputHeight * outputWidth;

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
    THCTensor_(resize4d)(state, gradOutput, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THCTensor_(resize4d)(state, input, nInputPlane, inputDepth, inputHeight, inputWidth);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

#endif
