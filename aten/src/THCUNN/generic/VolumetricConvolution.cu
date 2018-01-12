#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricConvolution.cu"
#else

static inline void THNN_(VolumetricConvolution_shapeCheck)
                        (THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         THCTensor *weight,
                         THCTensor *gradWeight,
                         THCTensor *bias,
                         int dT,
                         int dW,
                         int dH,
                         int padT,
                         int padW,
                         int padH) {
  THCUNN_argCheck(state, input->nDimension == 4 || input->nDimension == 5, 2, input,
                  "4D or 5D (batch mode) tensor expected for input, but got: %s");
  THArgCheck(!weight || THCTensor_(isContiguous)(state, weight), 4,
             "weight tensor has to be contiguous");
  THArgCheck(!bias || THCTensor_(isContiguous)(state, bias), 5,
             "bias tensor has to be contiguous");
  THArgCheck(!gradWeight || THCTensor_(isContiguous)(state, gradWeight), 5,
             "gradWeight tensor has to be contiguous");
  THArgCheck(dT > 0 && dW > 0 && dH > 0, 10,
             "stride should be greater than zero, but got dT: %d dH: %d dW: %d", dT, dH, dW);

  if (gradOutput != NULL) {
    THCUNN_argCheck(state, gradOutput->nDimension == 4 || gradOutput->nDimension == 5, 3,
                    gradOutput,
                    "4D or 5D (batch mode) tensor expected for gradOutput, but got: %s");
  }

  if (weight != NULL) {
    THCUNN_argCheck(state, weight->nDimension == 5, 4, weight,
                    "5D (nOutputPlane x nInputPlane x kT x kH x kW) tensor "
                    "expected for weight, but got: %s");
  }

  if (gradWeight != NULL) {
    THCUNN_argCheck(state, gradWeight->nDimension == 5, 4, gradWeight,
                    "5D (nOutputPlane x nInputPlane x kT x kH x kW) tensor "
                    "expected for gradWeight, but got: %s");
  }

  if (weight == NULL) {
    weight = gradWeight;
  }
  int nOutputPlane = (int)weight->size[0];
  int nInputPlane  = (int)weight->size[1];
  int kT           = (int)weight->size[2];
  int kH           = (int)weight->size[3];
  int kW           = (int)weight->size[4];

  THArgCheck(kT > 0 && kW > 0 && kH > 0, 4,
             "kernel size should be greater than zero, but got kT: %d kH: %d kW: %d", kT, kH, kW);
  int ndim = input->nDimension;
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;
  int dimd = 3;

  if (ndim == 5)
  {
    dimf++;
    dimh++;
    dimw++;
    dimd++;
  }

  int64_t inputWidth   = input->size[dimw];
  int64_t inputHeight  = input->size[dimh];
  int64_t inputDepth   = input->size[dimd];

  int64_t exactInputDepth = inputDepth + 2*padT;
  int64_t exactInputHeight = inputHeight + 2*padH;
  int64_t exactInputWidth = inputWidth + 2*padW;

  if (exactInputDepth < kT || exactInputHeight < kH || exactInputWidth < kW) {
    THError("Calculated input size: (%d x %d x %d). "
      "Kernel size: (%d x %d x %d). Kernel size can't greater than actual input size",
      exactInputDepth,exactInputHeight,exactInputWidth,kT,kH,kW);
  }

  int64_t outputWidth  = (exactInputDepth - kH) / dH + 1;
  int64_t outputHeight = (exactInputHeight - kT) / dT + 1;
  int64_t outputDepth  = (exactInputWidth - kW) / dW + 1;

  if (outputWidth < 1 || outputHeight < 1 || outputDepth < 1)
  {
    THError(
      "Given input size: (%dx%dx%dx%d). Calculated output size: (%dx%dx%dx%d). Output size is too small",
      nInputPlane, inputDepth, inputHeight, inputWidth,
      nOutputPlane, outputDepth, outputHeight, outputWidth
    );
  }

  if (bias != NULL) {
    THCUNN_check_dim_size(state, bias, 1, 0, weight->size[0]);
  }
  THCUNN_check_dim_size(state, input, ndim, dimf, nInputPlane);

  if (gradOutput != NULL) {
     THCUNN_check_dim_size(state, gradOutput, ndim, dimf, nOutputPlane);
     THCUNN_check_dim_size(state, gradOutput, ndim, dimh, outputHeight);
     THCUNN_check_dim_size(state, gradOutput, ndim, dimw, outputWidth);
     THCUNN_check_dim_size(state, gradOutput, ndim, dimd, outputDepth);
  }
}

void THNN_(VolumetricConvolution_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias,
           THCTensor *finput,
           THCTensor *fgradInput,
           int dT, int dW, int dH,
           int padT, int padW, int padH)
{
  THCTensor *columns = finput;
  THCTensor *ones = fgradInput;
  THCUNN_assertSameGPU(state, 6, input, output, weight, bias, columns, ones);
  THNN_(VolumetricConvolution_shapeCheck)(
        state, input, NULL, weight, NULL,
        bias, dT, dW, dH, padT, padW, padH);
  input = THCTensor_(newContiguous)(state, input);

  int nOutputPlane = (int)weight->size[0];
  int nInputPlane  = (int)weight->size[1];
  int kT           = (int)weight->size[2];
  int kH           = (int)weight->size[3];
  int kW           = (int)weight->size[4];

  int batch = 1;
  if (input->nDimension == 4)
  {
    // Force batch
    batch = 0;
    THCTensor_(resize5d)(state, input, 1, input->size[0], input->size[1],
                          input->size[2], input->size[3]);
  }

  int64_t inputWidth   = input->size[3];
  int64_t inputHeight  = input->size[2];
  int64_t inputDepth   = input->size[4];
  int64_t outputWidth  = (inputWidth  + 2*padH - kH) / dH + 1;
  int64_t outputHeight = (inputHeight + 2*padT - kT) / dT + 1;
  int64_t outputDepth  = (inputDepth  + 2*padW - kW) / dW + 1;

  // Batch size + input planes
  int64_t batchSize = input->size[0];

  // Resize output
  THCTensor_(resize5d)(state, output, batchSize, nOutputPlane,
                        outputHeight, outputWidth, outputDepth);

  // Resize temporary columns
  THCTensor_(resize2d)(state, columns, nInputPlane*kW*kH*kT, outputDepth*outputHeight*outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 3 || ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth)
  {
    // Resize plane and fill with ones...
    THCTensor_(resize3d)(state, ones, outputHeight, outputWidth, outputDepth);
    THCTensor_(fill)(state, ones, ScalarConvert<int, real>::to(1));
  }

  // Helpers
  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *output_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++)
  {
    // Matrix mulitply per output:
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    int64_t m_ = nOutputPlane;
    int64_t n_ = outputDepth * outputHeight * outputWidth;
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
    im3d2col(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input_n),
      nInputPlane, inputHeight, inputWidth, inputDepth, kT, kH, kW, padT, padH, padW, dT, dH, dW,
      THCTensor_(data)(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    int64_t m = weight->size[0];
    int64_t n = columns->size[1];
    int64_t k = weight->size[1]*weight->size[2]*weight->size[3]*weight->size[4];

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
  if (batch == 0)
  {
    THCTensor_(resize4d)(state, output, nOutputPlane, outputHeight, outputWidth, outputDepth);
    THCTensor_(resize4d)(state, input, nInputPlane, inputHeight, inputWidth, inputDepth);
  }
  THCTensor_(free)(state, input);
}

void THNN_(VolumetricConvolution_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *weight,
           THCTensor *finput,
           int dT, int dW, int dH,
           int padT, int padW, int padH)
{

  int nOutputPlane = (int)weight->size[0];
  int nInputPlane  = (int)weight->size[1];
  int kT           = (int)weight->size[2];
  int kH           = (int)weight->size[3];
  int kW           = (int)weight->size[4];

  THCTensor *gradColumns = finput;

  THCUNN_assertSameGPU(state, 5, input, gradOutput, weight, gradColumns, gradInput);
  THNN_(VolumetricConvolution_shapeCheck)(
        state, input, gradOutput, weight, NULL,
        NULL, dT, dW, dH, padT, padW, padH);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  int batch = 1;
  if (input->nDimension == 4)
  {
    input = THCTensor_(newContiguous)(state, input);
    // Force batch
    batch = 0;
    THCTensor_(resize5d)(state, input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THCTensor_(resize5d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  int64_t inputWidth   = input->size[3];
  int64_t inputHeight  = input->size[2];
  int64_t inputDepth   = input->size[4];
  int64_t outputWidth  = (inputWidth  + 2*padH - kH) / dH + 1;
  int64_t outputHeight = (inputHeight + 2*padT - kT) / dT + 1;
  int64_t outputDepth  = (inputDepth  + 2*padW - kW) / dW + 1;

  // Batch size + input planes
  int64_t batchSize = input->size[0];

  // Resize output
  THCTensor_(resize5d)(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth, inputDepth);

  // Resize temporary columns
  THCTensor_(resize2d)(state, gradColumns, nInputPlane*kH*kT*kW, outputDepth*outputHeight*outputWidth);

  // Helpers
  THCTensor *gradInput_n = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++)
  {
    // Matrix mulitply per sample:
    THCTensor_(select)(state, gradInput_n, gradInput, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    int64_t m = weight->size[1]*weight->size[2]*weight->size[3]*weight->size[4];
    int64_t n = gradColumns->size[1];
    int64_t k = weight->size[0];

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
    col2im3d<real, accreal>(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, gradColumns),
      nInputPlane, inputHeight, inputWidth, inputDepth, kT, kH, kW, padT, padH, padW, dT, dH, dW,
      THCTensor_(data)(state, gradInput_n)
    );
  }

  // Free
  THCTensor_(free)(state, gradInput_n);
  THCTensor_(free)(state, gradOutput_n);

  // Resize output
  if (batch == 0)
  {
    THCTensor_(resize4d)(state, gradOutput, nOutputPlane, outputHeight, outputWidth, outputDepth);
    THCTensor_(resize4d)(state, input, nInputPlane, inputHeight, inputWidth, inputDepth);
    THCTensor_(resize4d)(state, gradInput, nInputPlane, inputHeight, inputWidth, inputDepth);
    THCTensor_(free)(state, input);
  }
  THCTensor_(free)(state, gradOutput);

}

void THNN_(VolumetricConvolution_accGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *finput,
           THCTensor *fgradInput,
           int dT, int dW, int dH,
           int padT, int padW, int padH,
           accreal scale_)
{
  real scale = ScalarConvert<accreal, real>::to(scale_);
  THCTensor *columns = finput;
  THCTensor *ones = fgradInput;
  THCUNN_assertSameGPU(state, 6, input, gradOutput, gradWeight, gradBias, columns, ones);
  THNN_(VolumetricConvolution_shapeCheck)(
        state, input, gradOutput, NULL, gradWeight,
        gradBias, dT, dW, dH, padT, padW, padH);

  int nOutputPlane = (int)gradWeight->size[0];
  int nInputPlane  = (int)gradWeight->size[1];
  int kT           = (int)gradWeight->size[2];
  int kH           = (int)gradWeight->size[3];
  int kW           = (int)gradWeight->size[4];

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  int batch = 1;
  if (input->nDimension == 4)
  {
    // Force batch
    batch = 0;
    THCTensor_(resize5d)(state, input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THCTensor_(resize5d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  int64_t inputWidth   = input->size[3];
  int64_t inputHeight  = input->size[2];
  int64_t inputDepth   = input->size[4];
  int64_t outputWidth  = (inputWidth  + 2*padH - kH) / dH + 1;
  int64_t outputHeight = (inputHeight + 2*padT - kT) / dT + 1;
  int64_t outputDepth  = (inputDepth  + 2*padW - kW) / dW + 1;

  // Batch size + input planes
  int64_t batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 3 || ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth)
  {
    // Resize plane and fill with ones...
    THCTensor_(resize3d)(state, ones, outputHeight, outputWidth, outputDepth);
    THCTensor_(fill)(state, ones, ScalarConvert<int, real>::to(1));
  }

  // Resize temporary columns
  THCTensor_(resize2d)(state, columns, nInputPlane*kH*kT*kW, outputDepth*outputHeight*outputWidth);

  // Helpers
  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++)
  {
    // Matrix mulitply per output:
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    im3d2col(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input_n),
      nInputPlane, inputHeight, inputWidth, inputDepth, kT, kH, kW, padT, padH, padW, dT, dH, dW,
      THCTensor_(data)(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    int64_t m = gradWeight->size[0];
    int64_t n = gradWeight->size[1]*gradWeight->size[2]*gradWeight->size[3]*gradWeight->size[4];
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

    // Do Bias:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    int64_t m_ = nOutputPlane;
    int64_t k_ = outputDepth * outputHeight * outputWidth;

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
  if (batch == 0)
  {
    THCTensor_(resize4d)(state, gradOutput, nOutputPlane, outputHeight, outputWidth, outputDepth);
    THCTensor_(resize4d)(state, input, nInputPlane, inputHeight, inputWidth, inputDepth);
  }
  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

#endif
