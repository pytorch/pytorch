#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialDepthWiseConvolution.cu"
#else

static inline void THNN_(SpatialDepthWiseConvolution_shapeCheck)(
                         THCState *state,
                         THCTensor *input, THCTensor *gradOutput,
                         THCTensor *weight, THCTensor *bias,
                         int kH, int kW, int dH, int dW, int padH, int padW) {
  THArgCheck(kW > 0 && kH > 0, 9,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THCUNN_argCheck(state, weight->nDimension == 4, 5, weight,
                  "2D or 4D weight tensor expected, but got: %s");

  if (bias != NULL) {
    THCUNN_check_dim_size(state, bias, 2, 0, weight->size[0]);
    THCUNN_check_dim_size(state, bias, 2, 1, weight->size[1]);
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

  long nInputPlane  = weight->size[1];
  long inputHeight  = input->size[dimh];
  long inputWidth   = input->size[dimw];
  long nOutputPlane = weight->size[0];
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;

  if (outputWidth < 1 || outputHeight < 1)
      THError("Given input size: (%d x %d x %d). "
              "Calculated output size: (%d x %d x %d). Output size is too small",
              nInputPlane,inputHeight,inputWidth,nOutputPlane*nInputPlane,outputHeight,outputWidth);

  THCUNN_check_dim_size(state, input, ndim, dimf, nInputPlane);

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, ndim + 1, dimf, nInputPlane);
    THCUNN_check_dim_size(state, gradOutput, ndim + 1, dimh, nOutputPlane);
    THCUNN_check_dim_size(state, gradOutput, ndim + 1, dimw, outputHeight);
    THCUNN_check_dim_size(state, gradOutput, ndim + 1, dimw + 1, outputWidth);
  }
}

void THNN_(SpatialDepthWiseConvolution_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias,
           THCTensor *columns,
           THCTensor *ones,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) {

  THCUNN_assertSameGPU(state, 5, input, output, weight, columns, ones);
  if (bias) {
    THCUNN_assertSameGPU(state, 2, weight, bias);
  }

  // Params:
  int nInputPlane = weight->nDimension == 2 ? weight->size[1]/(kH*kW) : weight->size[1];
  int nOutputPlane = weight->size[0];
  if (weight->nDimension == 2) {
    THCTensor_(resize4d)(state, weight, nOutputPlane, nInputPlane, kH, kW);
  }

  THNN_(SpatialDepthWiseConvolution_shapeCheck)
       (state, input, NULL, weight, bias, kH, kW, dH, dW, padH, padW);


  // Transpose weight & bias
  THCTensor *_weight = THCTensor_(newTranspose)(state, weight, 0, 1);
  weight = THCTensor_(newContiguous)(state, _weight);

  THCTensor *_bias = NULL;
  if(bias) {
    _bias = THCTensor_(newTranspose)(state, bias, 0, 1);
    bias = THCTensor_(newContiguous)(state, _bias);
  }

  // resize weight
  long s1 = weight->size[0];
  long s2 = weight->size[1];
  long s3 = weight->size[2] * weight->size[3];
  weight = THCTensor_(newWithStorage3d)(state, weight->storage, weight->storageOffset,
          s1, -1, s2, -1, s3, -1);

  input = THCTensor_(newContiguous)(state, input);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCTensor_(resize5d)(state, output, batchSize, nInputPlane, nOutputPlane, outputHeight, outputWidth);

  // Resize temporary columns
  THCTensor_(resize2d)(state, columns, kW*kH, outputHeight*outputWidth);

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


  // Helpers for DepthWiseConvolution
  THCTensor *input_i = THCTensor_(new)(state);
  THCTensor *output_i = THCTensor_(new)(state);
  THCTensor *weight_i = THCTensor_(new)(state);

  THCTensor *bias_i = NULL;
  if(bias) {
    bias_i = THCTensor_(new)(state);
  }
  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);


    for (int ipelt = 0; ipelt < nInputPlane; ipelt++)
    {
      // Fetch ipelt-th input plane
      THCTensor_(narrow)(state, input_i, input_n, 0, ipelt, 1);
      THCTensor_(select)(state, output_i, output_n, 0, ipelt);
      THCTensor_(select)(state, weight_i, weight, 0, ipelt);
      if (bias) {
        THCTensor_(select)(state, bias_i, bias, 0, ipelt);
      }
      // Do Bias first:
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
            THCTensor_(data)(state, bias_i), k_,
            ScalarConvert<int, real>::to(0),
            THCTensor_(data)(state, output_i), n_
        );
      } else {
        THCTensor_(zero)(state, output_i);
      }

      // Extract columns:
      im2col(
        THCState_getCurrentStream(state),
        THCTensor_(data)(state, input_i),
        1, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
        1, 1, THCTensor_(data)(state, columns)
      );

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      long m = nOutputPlane;
      long n = columns->size[1];
      long k = 1*kH*kW;

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
          THCTensor_(data)(state, weight_i), k,
          ScalarConvert<int, real>::to(1),
          THCTensor_(data)(state, output_i), n
      );
    }
  }

  // Free
  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);

  THCTensor_(free)(state, input_i);
  THCTensor_(free)(state, output_i);

  THCTensor_(free)(state, weight_i);

  THCTensor_(free)(state, weight);
  THCTensor_(free)(state, _weight);

  THCTensor_(free)(state, bias_i);
  THCTensor_(free)(state, bias);
  THCTensor_(free)(state, _bias);
  // Transpose output
  THCTensor_(resize4d)(state, output, batchSize, nInputPlane * nOutputPlane, outputHeight, outputWidth);

  // Make a contiguous copy of output (OPTIONAL)
  // THCTensor *_output = THCTensor_(newContiguous)(state, output);

  // Resize output
  if (batch == 0) {
    THCTensor_(select)(state, output, NULL, 0, 0);
    THCTensor_(select)(state, input, NULL, 0, 0);
  }
  //else
    //THCTensor_(resize5d)(state, output, batchSize, nOutputPlane, nInputPlane, outputHeight, outputWidth);

  // Copy output back
  // THCTensor_(freeCopyTo)(state, _output, output);

  THCTensor_(free)(state, input);
}

void THNN_(SpatialDepthWiseConvolution_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *weight,
           THCTensor *gradColumns,
           THCTensor *ones,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) {

  THCUNN_assertSameGPU(state, 5, input, gradOutput, weight,
                       gradColumns, gradInput);

  // Params:
  int nInputPlane = weight->nDimension == 2 ? weight->size[1]/(kH*kW) : weight->size[1];
  int nOutputPlane = weight->size[0];
  if (weight->nDimension == 2) {
    THCTensor_(resize4d)(state, weight, nOutputPlane, nInputPlane, kH, kW);
  }

  gradOutput = THCTensor_(newWithTensor)(state, gradOutput);

  if (input->nDimension == 3) {
    if (gradOutput->nDimension == 3) {
      THCTensor_(resize4d)(state, gradOutput, nInputPlane, nOutputPlane, gradOutput->size[1], gradOutput->size[2]);
    }
  }
  else
  {
    if (gradOutput->nDimension == 4) {
      THCTensor_(resize5d)(state, gradOutput, gradOutput->size[0], nInputPlane, nOutputPlane, gradOutput->size[2], gradOutput->size[3]);
    }
  }

  THNN_(SpatialDepthWiseConvolution_shapeCheck)
       (state, input, gradOutput, weight, NULL, kH, kW, dH, dW, padH, padW);

  // Transpose weight
  THCTensor *_weight = THCTensor_(newTranspose)(state, weight, 0, 1);
  weight = THCTensor_(newContiguous)(state, _weight);

  // resize weight
  long s1 = weight->size[0];
  long s2 = weight->size[1];
  long s3 = weight->size[2] * weight->size[3];
  weight = THCTensor_(newWithStorage3d)(state, weight->storage, weight->storageOffset,
          s1, -1, s2, -1, s3, -1);



  input = THCTensor_(newContiguous)(state, input);


  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THCTensor_(resize5d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCTensor_(resize4d)(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  // Resize temporary columns
  THCTensor_(resize2d)(state, gradColumns, 1*kW*kH, outputHeight*outputWidth);

  // Helpers
  THCTensor *gradInput_n = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  // Helpers for DepthWiseConvolution
  THCTensor *gradOutput_i = THCTensor_(new)(state);
  THCTensor *gradInput_i = THCTensor_(new)(state);
  THCTensor *weight_i = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THCTensor_(select)(state, gradInput_n, gradInput, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    for (int ipelt = 0; ipelt < nInputPlane; ipelt++)
      {
      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)

      // Fetch ipelt-th input plane
      THCTensor_(narrow)(state, gradInput_i, gradInput_n, 0, ipelt, 1);
      THCTensor_(select)(state, gradOutput_i, gradOutput_n, 0, ipelt);
      THCTensor_(select)(state, weight_i, weight, 0, ipelt);

      long m = 1*kW*kH;
      long n = gradColumns->size[1];
      long k = nOutputPlane;

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
          THCTensor_(data)(state, gradOutput_i), n,
          THCTensor_(data)(state, weight_i), m,
          ScalarConvert<int, real>::to(0),
          THCTensor_(data)(state, gradColumns), n
      );

      // Unpack columns back into input:
      col2im<real, accreal>(
        THCState_getCurrentStream(state),
        THCTensor_(data)(state, gradColumns),
        1, inputHeight, inputWidth, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
        1, 1, THCTensor_(data)(state, gradInput_i)
      );
      }
  }

  // Free
  THCTensor_(free)(state, gradInput_n);
  THCTensor_(free)(state, gradOutput_n);

  THCTensor_(free)(state, gradInput_i);
  THCTensor_(free)(state, gradOutput_i);
  THCTensor_(free)(state, weight_i);

  // Resize output
  if (batch == 0) {
    THCTensor_(select)(state, gradOutput, NULL, 0, 0);
    THCTensor_(select)(state, input, NULL, 0, 0);
    THCTensor_(select)(state, gradInput, NULL, 0, 0);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
  THCTensor_(free)(state, weight);
  THCTensor_(free)(state, _weight);
}

void THNN_(SpatialDepthWiseConvolution_accGradParameters)(
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
           accreal scale_) {

  real scale = ScalarConvert<accreal, real>::to(scale_);

  THCUNN_assertSameGPU(state, 5, input, gradOutput, gradWeight, columns, ones);
  if (gradBias) {
   THCUNN_assertSameGPU(state, 2, gradWeight, gradBias);
  }

  // Params
  int nInputPlane = gradWeight->nDimension == 2 ? gradWeight->size[1]/(kW*kH) : gradWeight->size[1];
  int nOutputPlane = gradWeight->size[0];
  if (gradWeight->nDimension == 2) {
    THCTensor_(resize4d)(state, gradWeight, nOutputPlane, nInputPlane, kH, kW);
  }

 gradOutput = THCTensor_(newWithTensor)(state, gradOutput);
  if (input->nDimension == 3) {
    if (gradOutput->nDimension == 3) {
      THCTensor_(resize4d)(state, gradOutput, nInputPlane, nOutputPlane, gradOutput->size[1], gradOutput->size[2]);
    }
  }
  else
  {
    if (gradOutput->nDimension == 4) {
      THCTensor_(resize5d)(state, gradOutput, gradOutput->size[0], nInputPlane, nOutputPlane, gradOutput->size[2], gradOutput->size[3]);
    }
  }


  THNN_(SpatialDepthWiseConvolution_shapeCheck)
       (state, input, gradOutput, gradWeight, gradBias, kH, kW, dH, dW, padH, padW);

  // Transpose gradWeight & gradBias
  THCTensor_(transpose)(state, gradWeight, NULL, 0, 1);


  THCTensor *_gradBias = NULL;
  if(gradBias) {
    THCTensor_(transpose)(state, gradBias, NULL, 0, 1);
    _gradBias = gradBias;
    gradBias = THCTensor_(newContiguous)(state, gradBias);

  }

  THCTensor *_gradWeight;

  _gradWeight = gradWeight;

  gradWeight = THCTensor_(newContiguous)(state, gradWeight);


  // resize gradWeight
  long s1 = gradWeight->size[0];
  long s2 = gradWeight->size[1];
  long s3 = gradWeight->size[2] * gradWeight->size[3];
  gradWeight = THCTensor_(newWithStorage3d)(state, gradWeight->storage, gradWeight->storageOffset,
          s1, -1, s2, -1, s3, -1);

  input = THCTensor_(newContiguous)(state, input);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THCTensor_(resize5d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCTensor_(resize2d)(state, ones, outputHeight, outputWidth);
    THCTensor_(fill)(state, ones, ScalarConvert<int, real>::to(1));
  }

  // Resize temporary columns
  THCTensor_(resize2d)(state, columns, 1*kW*kH, outputHeight*outputWidth);

  // Helpers
  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  // Helpers for DepthWiseConvolution
  THCTensor *gradOutput_i = THCTensor_(new)(state);
  THCTensor *input_i = THCTensor_(new)(state);
  THCTensor *gradWeight_i = THCTensor_(new)(state);

  THCTensor *gradBias_i = NULL;
  if(gradBias) {
    gradBias_i = THCTensor_(new)(state);
  }

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    for (int ipelt = 0; ipelt < nInputPlane; ipelt++)
    {
      THCTensor_(narrow)(state, input_i, input_n, 0, ipelt, 1);
      THCTensor_(select)(state, gradOutput_i, gradOutput_n, 0, ipelt);
      THCTensor_(select)(state, gradWeight_i, gradWeight, 0, ipelt);
      if (gradBias) {
        THCTensor_(select)(state, gradBias_i, gradBias, 0, ipelt);
      }

      // Extract columns:
      im2col(
        THCState_getCurrentStream(state),
        THCTensor_(data)(state, input_i),
        1, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
        1, 1, THCTensor_(data)(state, columns)
      );

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      long m = nOutputPlane;
      long n = 1*kW*kH;
      long k = columns->size[1];

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
          THCTensor_(data)(state, gradOutput_i), k,
          ScalarConvert<int, real>::to(1),
          THCTensor_(data)(state, gradWeight_i), n
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
            THCTensor_(data)(state, gradOutput_i), k_,
            THCTensor_(data)(state, ones), 1,
            ScalarConvert<int, real>::to(1),
            THCTensor_(data)(state, gradBias_i), 1
        );
        #endif
        #ifdef THC_REAL_IS_HALF
        THCudaBlas_Hgemm(
            state,
            't', 'n',
            m_, 1, k_,
            scale,
            THCTensor_(data)(state, gradOutput_i), k_,
            THCTensor_(data)(state, ones), k_,
            ScalarConvert<int, real>::to(1),
            THCTensor_(data)(state, gradBias_i), m_
        );
        #endif
      }
    }
  }


  // Copy back and transpose back
  THCTensor_(transpose)(state, _gradWeight, NULL, 0, 1);
  THCTensor_(resize4d)(state, _gradWeight, nInputPlane, nOutputPlane, kH, kW);
  THCTensor_(copy)(state, _gradWeight, gradWeight);
  THCTensor_(transpose)(state, _gradWeight, NULL, 0, 1);

  if(gradBias) {
    THCTensor_(transpose)(state, _gradBias, NULL, 0, 1);
    THCTensor_(resize2d)(state, _gradBias, nInputPlane, nOutputPlane);
    THCTensor_(copy)(state, _gradBias, gradBias);
    THCTensor_(transpose)(state, _gradBias, NULL, 0, 1);
  }


  // Free
  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, gradOutput_n);
  THCTensor_(free)(state, input_i);
  THCTensor_(free)(state, gradOutput_i);
  THCTensor_(free)(state, gradWeight_i);
  THCTensor_(free)(state, gradWeight);
  THCTensor_(free)(state, gradBias_i);
  THCTensor_(free)(state, gradBias);

  // Resize
  if (batch == 0) {
    THCTensor_(select)(state, gradOutput, NULL, 0, 0);
    THCTensor_(select)(state, input, NULL, 0, 0);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

#endif
