#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialFullConvolution.c"
#else

static void THNN_(im2col)(const real* data_im, const int channels,
      const int height, const int width, const int kernel_h, const int kernel_w,
      const int pad_h, const int pad_w,
      const int stride_h, const int stride_w,
      const int dilation_h, const int dilation_w,
      real* data_col) {
  const int height_col = (height + 2 * pad_h -
                          (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_col = (width + 2 * pad_w -
                         (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channels_col = channels * kernel_h * kernel_w;
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    int w_offset = c_col % kernel_w;
    int h_offset = (c_col / kernel_w) % kernel_h;
    int c_im = c_col / kernel_h / kernel_w;
    for (int h_col = 0; h_col < height_col; ++h_col) {
      for (int w_col = 0; w_col < width_col; ++w_col) {
        int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
        int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
        data_col[(c_col * height_col + h_col) * width_col + w_col] =
          (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
          data_im[(c_im * height + h_im) * width + w_im] : 0;
      }
    }
  }
}

static void THNN_(col2im)(const real* data_col, const int channels,
      const int height, const int width, const int kernel_h, const int kernel_w,
      const int pad_h, const int pad_w,
      const int stride_h, const int stride_w,
      const int dilation_h, const int dilation_w,
      real* data_im) {
  memset(data_im, 0, sizeof(real) * height * width * channels);
  const int height_col = (height + 2 * pad_h -
                          (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_col = (width + 2 * pad_w -
                         (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channels_col = channels * kernel_h * kernel_w;
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    int w_offset = c_col % kernel_w;
    int h_offset = (c_col / kernel_w) % kernel_h;
    int c_im = c_col / kernel_h / kernel_w;
    for (int h_col = 0; h_col < height_col; ++h_col) {
      for (int w_col = 0; w_col < width_col; ++w_col) {
        int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
        int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
        if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
          data_im[(c_im * height + h_im) * width + w_im] +=
            data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
  }
}

static inline void THNN_(SpatialFullConvolution_shapeCheck)(
	THTensor *input, THTensor *gradOutput,
	THTensor *weight, THTensor *bias, 
	int kH, int kW, int dH, int dW, int padH, int padW, int adjH, int adjW) {

  THArgCheck(kW > 0 && kH > 0, 9,
	       "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 11,
	     "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THNN_ARGCHECK(weight->nDimension == 2 || weight->nDimension == 4, 5, weight,
		"2D or 4D weight tensor expected, but got: %s");

  if (bias != NULL) {
    THNN_CHECK_DIM_SIZE(bias, 1, 0, weight->size[1]);
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

  long nInputPlane  = weight->size[0];
  long inputHeight  = input->size[dimh];
  long inputWidth   = input->size[dimw];
  long nOutputPlane = weight->size[1];
  long outputHeight = (inputHeight - 1) * dH - 2*padH + kH + adjH;
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + kW + adjW;

  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%d x %d x %d). "
	    "Calculated output size: (%d x %d x %d). Output size is too small",
	    nInputPlane,inputHeight,inputWidth,nOutputPlane,outputHeight,outputWidth);

  THNN_CHECK_DIM_SIZE(input, ndim, dimf, nInputPlane);
  
  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
}

void THNN_(SpatialFullConvolution_updateOutput)(
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
    int adjW, int adjH)
{
  THNN_(SpatialFullConvolution_shapeCheck)
    (input, NULL, weight, bias, kH, kW, dH, dW, padH, padW, adjH, adjW);

  int nInputPlane = THTensor_(size)(weight,0);
  int nOutputPlane = THTensor_(size)(weight,1);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
  }

  long inputHeight  = input->size[2];
  long inputWidth   = input->size[3];
  long outputHeight = (inputHeight - 1) * dH - 2*padH + kH + adjH;
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + kW + adjW;

  // Batch size + input planes
  long batchSize = input->size[0];

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
    long m = weight->size[1] * weight->size[2] * weight->size[3];
    long n = columns->size[1];
    long k = weight->size[0];

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
      nOutputPlane, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
      1, 1,
      THTensor_(data)(output_n)
    );

    // Do Bias after:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
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
          1,
          THTensor_(data)(output_n), n_
      );
    }
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

void THNN_(SpatialFullConvolution_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    THTensor *weight,
    THTensor *gradColumns,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int adjW, int adjH)
{
  THNN_(SpatialFullConvolution_shapeCheck)
    (input, gradOutput, weight, NULL, kH, kW, dH, dW, padH, padW, adjH, adjW);

  int nInputPlane = THTensor_(size)(weight,0);
  int nOutputPlane = THTensor_(size)(weight,1);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
    THTensor_(resize4d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + kW + adjW;
  long outputHeight = (inputHeight - 1) * dH - 2*padH + kH + adjH;

  // Batch size + input planes
  long batchSize = input->size[0];

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
      nOutputPlane, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
      1, 1,
      THTensor_(data)(gradColumns)
    );


    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[0];
    long n = gradColumns->size[1];
    long k = weight->size[1] * weight->size[2] * weight->size[3];

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
  if (batch == 0) {
    THTensor_(resize3d)(gradOutput, nOutputPlane, outputHeight, outputWidth);
    THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
    THTensor_(resize3d)(gradInput, nInputPlane, inputHeight, inputWidth);
  }
}


void THNN_(SpatialFullConvolution_accGradParameters)(
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
    int adjW, int adjH,
    real scale)
{
  THNN_(SpatialFullConvolution_shapeCheck)
    (input, gradOutput, gradWeight, gradBias, kH, kW, dH, dW, padH, padW, adjH, adjW);

  int nInputPlane = THTensor_(size)(gradWeight,0);
  int nOutputPlane = THTensor_(size)(gradWeight,1);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
    THTensor_(resize4d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + kW + adjW;
  long outputHeight = (inputHeight - 1) * dH - 2*padH + kH + adjH;

  // Batch size + input planes
  long batchSize = input->size[0];

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
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    THNN_(im2col)(
      THTensor_(data)(gradOutput_n),
      nOutputPlane, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
      1, 1,
      THTensor_(data)(columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long n = columns->size[0];   // nOutputPlane * kh * kw
    long m = input_n->size[0];   // nInputPlane
    long k = columns->size[1];   // inputHeight * inputWidth

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


    // Do Bias:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
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
