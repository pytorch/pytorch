#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/IndexedConvolution.cu"
#else

static inline void THNN_(IndexedConvolution_shapeCheck)(
                         THCState *state,
                         THCTensor *input, THCTensor *gradOutput,
                         THCTensor *weight, THCTensor *bias,
                         THCIndexTensor *indices) {

  THCUNN_argCheck(state, weight->nDimension == 3, 3, weight,
		              "3D weight tensor (nOutputPlane,nInputPlane,kernelSize) expected, "
		              "but got: %s");
  THArgCheck(THCTensor_(isContiguous)(state, weight), 4,
             "weight tensor has to be contiguous");
  THArgCheck(!bias || THCTensor_(isContiguous)(state, bias), 5,
             "bias tensor has to be contiguous");

  if (bias != NULL) {
    THCUNN_check_dim_size(state, bias, 1, 0, weight->size[0]);
  }

  int ndim = input->nDimension;
  int dimf = 0;
  int dimw = 1;

  if (ndim == 3) {
    dimf++;
    dimw++;
  }

  THCUNN_argCheck(state, ndim == 2 || ndim == 3, 2, input,
                  "2D or 3D input tensor expected but got: %s");

  int64_t inputWidth   = input->size[dimw];
  int64_t nOutputPlane = weight->size[0];
  int64_t nInputPlane  = weight->size[1];
  int64_t kSize  = weight->size[2];
  // outputWidth will change when strides are introduced
  int64_t outputWidth  = inputWidth;

  if (outputWidth < 1)
    THError("Given input size: (%ld x %ld). "
            "Calculated output size: (%ld x %ld). Output size is too small",
            nInputPlane,inputWidth,nOutputPlane,outputWidth);

  THCUNN_check_dim_size(state, input, ndim, dimf, nInputPlane);

  THCUNN_check_dim_size_indices(state, indices, 2, 0, inputWidth);
  THCUNN_check_dim_size_indices(state, indices, 2, 1, kSize);

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, ndim, dimf, nOutputPlane);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimw, outputWidth);
  }
}

void THNN_(IndexedConvolution_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias,
           THCIndexTensor *indices,
           THCTensor *columns,
           THCTensor *ones) {

  THCUNN_assertSameGPU(state, 6, input, output, weight, indices, columns, ones);
  if (bias) {
    THCUNN_assertSameGPU(state, 2, weight, bias);
  }
  THNN_(IndexedConvolution_shapeCheck)
       (state, input, NULL, weight, bias, indices);

  // Params:
  int nInputPlane = weight->size[1];
  int nOutputPlane = weight->size[0];
  int kSize = indices->size[1];

  input = THCTensor_(newContiguous)(state, input);
  weight = THCTensor_(newContiguous)(state, weight);
  bias = bias ? THCTensor_(newContiguous)(state, bias) : bias;
  indices = THCudaLongTensor_newContiguous(state, indices);
  
  int batch = 1;
  if (input->nDimension == 2) {
    // Force batch
    batch = 0;
    THCTensor_(resize3d)(state, input, 1, input->size[0], input->size[1]);
  }

  int64_t inputWidth   = input->size[2];
  // outputWidth will change when strides are introduced
  int64_t outputWidth = inputWidth;

  // Batch size + input planes
  int64_t batchSize = input->size[0];

  // Resize output
  THCTensor_(resize3d)(state, output, batchSize, nOutputPlane, outputWidth);

  // Resize temporary columns
  THCTensor_(resize2d)(state, columns, nInputPlane*kSize, outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 1 || ones->size[0] < outputWidth) {
    // Resize plane and fill with ones...
    THCTensor_(resize1d)(state, ones, outputWidth);
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
    int64_t n_ = outputWidth;
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
    idx2col(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input_n),
      nInputPlane, inputWidth, kSize,
      THCudaLongTensor_data(state, indices),
      THCTensor_(data)(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    int64_t m = nOutputPlane;
    int64_t n = columns->size[1];
    int64_t k = nInputPlane*kSize;

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
  if (batch == 0) {
    THCTensor_(resize2d)(state, output, nOutputPlane, outputWidth);
    THCTensor_(resize2d)(state, input, nInputPlane, inputWidth);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, weight);
  if (bias) THCTensor_(free)(state, bias);
  THCudaLongTensor_free(state, indices);
}

void THNN_(IndexedConvolution_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *weight,
           THCIndexTensor *indices,
           THCTensor *gradColumns) {

  THCUNN_assertSameGPU(state, 6, input, gradOutput, weight, indices,
                       gradColumns, gradInput);
  THNN_(IndexedConvolution_shapeCheck)
       (state, input, gradOutput, weight, NULL, indices);

  // Params
  int nInputPlane = weight->size[1];
  int nOutputPlane = weight->size[0];
  int kSize = indices->size[1];

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  weight = THCTensor_(newContiguous)(state, weight);
  indices = THCudaLongTensor_newContiguous(state, indices);

  int batch = 1;
  if (input->nDimension == 2) {
    // Force batch
    batch = 0;
    THCTensor_(resize3d)(state, input, 1, input->size[0], input->size[1]);
    THCTensor_(resize3d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1]);
  }

  int64_t inputWidth   = input->size[2];
  //int64_t outputWidth  = (inputWidth + 2*padW - (dilationW * (kSize - 1) + 1)) / dW + 1;
  int64_t outputWidth  = inputWidth;

  // Batch size + input planes
  int64_t batchSize = input->size[0];

  // Resize output
  THCTensor_(resize3d)(state, gradInput, batchSize, nInputPlane, inputWidth);
  THCTensor_(zero)(state, gradInput);

  // Resize temporary columns
  THCTensor_(resize2d)(state, gradColumns, nInputPlane*kSize, outputWidth);
  THCTensor_(zero)(state, gradColumns);

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
    int64_t m = nInputPlane*kSize;
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
    col2idx<real, accreal>(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, gradColumns),
      nInputPlane, inputWidth, kSize,
      THCudaLongTensor_data(state, indices),
      THCTensor_(data)(state, gradInput_n)
    );
  }

  // Free
  THCTensor_(free)(state, gradInput_n);
  THCTensor_(free)(state, gradOutput_n);

  // Resize output
  if (batch == 0) {
    THCTensor_(resize2d)(state, gradOutput, nOutputPlane, outputWidth);
    THCTensor_(resize2d)(state, input, nInputPlane, inputWidth);
    THCTensor_(resize2d)(state, gradInput, nInputPlane, inputWidth);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
  THCTensor_(free)(state, weight);
  THCudaLongTensor_free(state, indices);
}

void THNN_(IndexedConvolution_accGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCIndexTensor *indices,
           THCTensor *columns,
           THCTensor *ones,
           accreal scale_) {

  real scale = ScalarConvert<accreal, real>::to(scale_);
  THCUNN_assertSameGPU(state, 6, input, gradOutput, gradWeight, indices, columns, ones);
  if (gradBias) {
   THCUNN_assertSameGPU(state, 2, gradWeight, gradBias);
  }
  THNN_(IndexedConvolution_shapeCheck)
       (state, input, gradOutput, gradWeight, gradBias, indices);

  THArgCheck(THCTensor_(isContiguous)(state, gradWeight), 4, "gradWeight needs to be contiguous");
  if (gradBias)
    THArgCheck(THCTensor_(isContiguous)(state, gradBias), 5, "gradBias needs to be contiguous");
  
  // Params
  int nInputPlane = gradWeight->size[1];
  int nOutputPlane = gradWeight->size[0];
  int kSize = indices->size[1];

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  indices = THCudaLongTensor_newContiguous(state, indices);
  int batch = 1;
  if (input->nDimension == 2) {
    // Force batch
    batch = 0;
    THCTensor_(resize3d)(state, input, 1, input->size[0], input->size[1]);
    THCTensor_(resize3d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1]);
  }

  int64_t inputWidth   = input->size[2];
  int64_t outputWidth  = inputWidth;
  //int64_t outputWidth  = (inputWidth + 2*padW - (dilationW * (kSize - 1) + 1)) / dW + 1;

  // Batch size + input planes
  int64_t batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 1 || ones->size[0] < outputWidth) {
    // Resize plane and fill with ones...
    THCTensor_(resize1d)(state, ones, outputWidth);
    THCTensor_(fill)(state, ones, ScalarConvert<int, real>::to(1));
  }

  // Resize temporary columns
  THCTensor_(resize2d)(state, columns, nInputPlane*kSize, outputWidth);

  // Helpers
  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    idx2col(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input_n),
      nInputPlane, inputWidth, kSize,
      THCudaLongTensor_data(state, indices),
      THCTensor_(data)(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    int64_t m = nOutputPlane;
    int64_t n = nInputPlane*kSize;
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
    int64_t k_ = outputWidth;

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
    THCTensor_(resize2d)(state, gradOutput, nOutputPlane, outputWidth);
    THCTensor_(resize2d)(state, input, nInputPlane, inputWidth);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
  THCudaLongTensor_free(state, indices);
}

#endif
