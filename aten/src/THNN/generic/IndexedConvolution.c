#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/IndexedConvolution.c"
#else

// Note: strides could be specified using a mask tensor

static void THNN_(idx2col)(const real* data_im, const int channels,
      const int width,
      const int kernel_size,
      const int64_t* data_idx,
      real* data_col) {

  // width_col will change when strides are introduced
  const int width_col = width;
  const int channels_col = channels * kernel_size;

  for (int c_col = 0; c_col < channels_col; ++c_col) {

    int c_im = c_col / kernel_size;
    int k_el = c_col % kernel_size;

    for (int w_col = 0; w_col < width_col; ++w_col) {

      int w_im = data_idx[w_col * kernel_size + k_el];

      data_col[c_col * width_col + w_col] =
        (w_im >= 0 && w_im < width) ?
        data_im[c_im * width + w_im] : 0;
    }
  }
}

static void THNN_(col2idx)(const real* data_col, const int channels,
      const int width,
      const int output_width,
      const int kernel_size,
      const int64_t* data_idx,
      real* data_im) {

  memset(data_im, 0, sizeof(real) * width * channels);

  const int width_col = output_width;
  const int channels_col = channels * kernel_size;

  for (int c_col = 0; c_col < channels_col; ++c_col) {

    int c_im = c_col / kernel_size;
    int k_el = c_col % kernel_size;

    for (int w_col = 0; w_col < width_col; ++w_col) {

      int w_im = data_idx[w_col * kernel_size + k_el];

      if (w_im >= 0 && w_im < width)
        data_im[c_im * width + w_im] +=
          data_col[c_col * width_col + w_col];
    }
  }
}

static inline void THNN_(IndexedConvolution_shapeCheck)(
	THTensor *input, THTensor *gradOutput,
	THTensor *weight, THTensor *bias,
  THIndexTensor *indices) {

  THNN_ARGCHECK(weight->nDimension == 3, 3, weight,
                "3D weight tensor (nOutputPlane,nInputPlane,kSize) expected, "
                "but got: %s");

  if (bias != NULL) {
    THNN_CHECK_DIM_SIZE(bias, 1, 0, weight->size[0]);
  }

  int ndim = input->nDimension;
  int dimf = 0;
  int dimw = 1;

  if (ndim == 3) {
    dimf++;
    dimw++;
  }

  THNN_ARGCHECK(ndim == 2 || ndim == 3, 2, input,
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
	    nInputPlane, inputWidth, nOutputPlane, outputWidth);

  THNN_CHECK_DIM_SIZE(input, ndim, dimf, nInputPlane);

  THNN_CHECK_DIM_SIZE_INDICES(indices, 2, 0, inputWidth);
  THNN_CHECK_DIM_SIZE_INDICES(indices, 2, 1, kSize);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
}

void THNN_(IndexedConvolution_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    THTensor *weight,
    THTensor *bias,
    THIndexTensor *indices,
    THTensor *columns,
    THTensor *ones)
{
  THNN_(IndexedConvolution_shapeCheck)
    (input, NULL, weight, bias, indices);

  // Params:
  int nInputPlane = weight->size[1];
  int nOutputPlane = weight->size[0];
  int kSize = indices->size[1];

  input = THTensor_(newContiguous)(input);
  weight = THTensor_(newContiguous)(weight);
  bias = bias ? THTensor_(newContiguous)(bias) : bias;
  indices = THLongTensor_newContiguous(indices);

  int batch = 1;
  if (input->nDimension == 2) {
    // Force batch
    batch = 0;
    THTensor_(resize3d)(input, 1, input->size[0], input->size[1]);
  }

  int64_t inputWidth   = input->size[2];
  // outputWidth will change when strides are introduced
  int64_t outputWidth  = inputWidth;

  // Batch size + input planes
  int64_t batchSize = input->size[0];

  // Resize output
  THTensor_(resize3d)(output, batchSize, nOutputPlane, outputWidth);
  THTensor_(zero)(output);

  // Resize temporary columns
  THTensor_(resize2d)(columns, nInputPlane*kSize, outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 1 || ones->size[0] < outputWidth) {
    // Resize plane and fill with ones...
    THTensor_(resize1d)(ones, outputWidth);
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
    int64_t n_ = outputWidth;
    int64_t k_ = 1;

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
    THNN_(idx2col)(
      THTensor_(data)(input_n),
      nInputPlane, inputWidth, kSize,
      THLongTensor_data(indices),
      THTensor_(data)(columns)
    );

    // M,N,K are dims of matrix A and B
    int64_t m = nOutputPlane;
    int64_t n = columns->size[1];
    int64_t k = nInputPlane*kSize;

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
    THTensor_(resize2d)(output, nOutputPlane, outputWidth);
    THTensor_(resize2d)(input, nInputPlane, inputWidth);
  }

  THTensor_(free)(input);
  THTensor_(free)(weight);
  if (bias) THTensor_(free)(bias);
  THLongTensor_free(indices);
}

void THNN_(IndexedConvolution_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    THTensor *weight,
    THIndexTensor *indices,
    THTensor *gradColumns)
{
  THNN_(IndexedConvolution_shapeCheck)
    (input, gradOutput, weight, NULL, indices);

  // Params
  int nInputPlane = weight->size[1];
  int nOutputPlane = weight->size[0];
  int kSize = indices->size[1];

  input = THTensor_(newContiguous)(input);
  weight = THTensor_(newContiguous)(weight);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  indices = THLongTensor_newContiguous(indices);

  int batch = 1;
  if (input->nDimension == 2) {
    // Force batch
    batch = 0;
    THTensor_(resize3d)(input, 1, input->size[0], input->size[1]);
    THTensor_(resize3d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1]);
  }

  int64_t inputWidth   = input->size[2];
  int64_t outputWidth  = inputWidth;

  // Batch size + input planes
  int64_t batchSize = input->size[0];

  // Resize output
  THTensor_(resize3d)(gradInput, batchSize, nInputPlane, inputWidth);

  // Resize temporary columns
  THTensor_(resize2d)(gradColumns, nInputPlane*kSize, outputWidth);
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
    int64_t m = nInputPlane*kSize;
    int64_t n = gradColumns->size[1];
    int64_t k = nOutputPlane;

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
    THNN_(col2idx)(
      THTensor_(data)(gradColumns),
      nInputPlane, inputWidth, outputWidth,
      kSize,
      THLongTensor_data(indices),
      THTensor_(data)(gradInput_n)
    );
  }

  // Free
  THTensor_(free)(gradInput_n);
  THTensor_(free)(gradOutput_n);

  // Resize output
  if (batch == 0) {
    THTensor_(resize2d)(gradOutput, nOutputPlane, outputWidth);
    THTensor_(resize2d)(input, nInputPlane, inputWidth);
    THTensor_(resize2d)(gradInput, nInputPlane, inputWidth);
  }

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
  THTensor_(free)(weight);
  THLongTensor_free(indices);
}

void THNN_(IndexedConvolution_accGradParameters)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradWeight,
    THTensor *gradBias,
    THIndexTensor *indices,
    THTensor *columns,
    THTensor *ones,
    accreal scale_)
{
  real scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  THNN_(IndexedConvolution_shapeCheck)
    (input, gradOutput, gradWeight, gradBias, indices);

  // Params
  int nInputPlane = gradWeight->size[1];
  int nOutputPlane = gradWeight->size[0];
  int kSize = indices->size[1];

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  indices = THLongTensor_newContiguous(indices);
  THArgCheck(THTensor_(isContiguous)(gradWeight), 4, "gradWeight needs to be contiguous");
  if (gradBias)
    THArgCheck(THTensor_(isContiguous)(gradBias), 5, "gradBias needs to be contiguous");
  int batch = 1;
  if (input->nDimension == 2) {
    // Force batch
    batch = 0;
    THTensor_(resize3d)(input, 1, input->size[0], input->size[1]);
    THTensor_(resize3d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1]);
  }

  int64_t inputWidth   = input->size[2];
  int64_t outputWidth  = inputWidth;

  // Batch size + input planes
  int64_t batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 1 || ones->size[0] < outputWidth) {
    // Resize plane and fill with ones...
    THTensor_(resize1d)(ones, outputWidth);
    THTensor_(fill)(ones, 1);
  }

  // Resize temporary columns
  THTensor_(resize2d)(columns, nInputPlane*kSize, outputWidth);

  // Helpers
  THTensor *input_n = THTensor_(new)();
  THTensor *gradOutput_n = THTensor_(new)();

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    THNN_(idx2col)(
      THTensor_(data)(input_n),
      nInputPlane, inputWidth, kSize,
      THLongTensor_data(indices),
      THTensor_(data)(columns)
    );

    // M,N,K are dims of matrix A and B
    int64_t m = nOutputPlane;
    int64_t n = nInputPlane*kSize;
    int64_t k = columns->size[1];

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
    int64_t m_ = nOutputPlane;
    int64_t k_ = outputWidth;

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
    THTensor_(resize2d)(gradOutput, nOutputPlane, outputWidth);
    THTensor_(resize2d)(input, nInputPlane, inputWidth);
  }

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
  THLongTensor_free(indices);
}

#endif
