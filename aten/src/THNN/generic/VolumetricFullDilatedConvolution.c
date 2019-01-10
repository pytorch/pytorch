#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricFullDilatedConvolution.c"
#else

static void THNN_(vol2col)(
  const real *data_vol, const int channels,
  const int depth, const int height, const int width,
  const int depth_col, const int height_col, const int width_col,
  const int kT, const int kH, const int kW,
  const int pT, const int pH, const int pW,
  const int dT, const int dH, const int dW,
  const int dilationT, const int dilationH, const int dilationW,
  real *data_col)
{
  int c, t, h, w;
  int channels_col = channels * kT * kH * kW;
  for (c = 0; c < channels_col; ++c)
  {
    int w_offset = c % kW;
    int h_offset = (c / kW) % kH;
    int t_offset = (c / kW / kH) % kT;
    int c_vol = c / kT / kH / kW;
    for (t = 0; t < depth_col; ++t)
    {
      for (h = 0; h < height_col; ++h)
      {
        for (w = 0; w < width_col; ++w)
        {
          int t_pad = t * dT - pT + t_offset * dilationT;
          int h_pad = h * dH - pH + h_offset * dilationH;
          int w_pad = w * dW - pW + w_offset * dilationW;
          if (t_pad >= 0 && t_pad < depth &&
              h_pad >= 0 && h_pad < height &&
              w_pad >= 0 && w_pad < width)
            data_col[((c * depth_col + t) * height_col + h) * width_col + w] =
              data_vol[((c_vol * depth + t_pad) * height + h_pad) * width + w_pad];
          else
            data_col[((c * depth_col + t) * height_col + h) * width_col + w] = 0;
        }
      }
    }
  }
}

static void THNN_(col2vol)(
  const real* data_col, const int channels,
  const int depth, const int height, const int width,
  const int out_depth, const int out_height, const int out_width,
  const int kT, const int kH, const int kW,
  const int pT, const int pH, const int pW,
  const int dT, const int dH, const int dW,
  const int dilationT, const int dilationH, const int dilationW,
  real* data_vol)
{
  int c, t, h, w;
  memset(data_vol, 0, sizeof(real) * depth * height * width * channels);
  int depth_col  = out_depth;
  int height_col = out_height;
  int width_col  = out_width;
  int channels_col = channels * kT * kH * kW;
  for (c = 0; c < channels_col; ++c)
  {
    int w_offset = c % kW;
    int h_offset = (c / kW) % kH;
    int t_offset = (c / kW / kH) % kT;
    int c_vol = c / kT / kH / kW;
    for (t = 0; t < depth_col; ++t)
    {
      for (h = 0; h < height_col; ++h)
      {
        for (w = 0; w < width_col; ++w)
        {
          int t_pad = t * dT - pT + t_offset * dilationT;
          int h_pad = h * dH - pH + h_offset * dilationH;
          int w_pad = w * dW - pW + w_offset * dilationW;
          if (t_pad >= 0 && t_pad < depth &&
              h_pad >= 0 && h_pad < height &&
              w_pad >= 0 && w_pad < width)
            data_vol[((c_vol * depth + t_pad) * height + h_pad) * width + w_pad] +=
              data_col[((c * depth_col + t) * height_col + h) * width_col + w];
        }
      }
    }
  }
}

static inline void THNN_(VolumetricFullDilatedConvolution_shapeCheck)(
                         THTensor *input, THTensor *gradOutput,
                         THTensor *weight, THTensor *bias,
                         int kT, int kW, int kH, int dT, int dW, int dH,
                         int pT, int pW, int pH,
                         int dilationT, int dilationW, int dilationH,
                         int aT, int aW, int aH, int weight_nullable) {
  THNN_ARGCHECK(input->nDimension == 4 || input->nDimension == 5, 2, input,
                "4D or 5D (batch mode) tensor expected for input, but got: %s");
  THArgCheck(dT > 0 && dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dT: %d dH: %d dW: %d", dT, dH, dW);
  THArgCheck(dilationT > 0 && dilationW > 0 && dilationH > 0, 15,
             "dilation should be greater than zero, but got dilationT: %d, dilationH: %d, dilationW: %d",
             dilationT, dilationH, dilationW);
  THArgCheck((aT < dT || aT < dilationT)
             && (aW < dW || aW < dilationW)
             && (aH < dH || aH < dilationH), 15,
             "output padding must be smaller than either stride or dilation,"
             " but got aT: %d aH: %d aW: %d dT: %d dH: %d dW: %d "
             "dilationT: %d dilationH: %d dilationW: %d",
             aT, aH, aW, dT, dH, dW, dilationT, dilationH, dilationW);

  // number of input & output planes and kernel size is indirectly defined by the weight tensor
  if (weight != NULL) {
    THNN_ARGCHECK(weight->nDimension == 5, 4, weight,
                  "5D (nOutputPlane x nInputPlane x kT x kH x kW) tensor "
                  "expected for weight, but got: %s");
    if (bias != NULL) {
      THNN_CHECK_DIM_SIZE(bias, 1, 0, weight->size[1]);
    }
  } else if (!weight_nullable) {
    THError("weight tensor is expected to be non-nullable");
  }

  int ndim = input->nDimension;
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

  if (weight != NULL) {
    const int64_t nInputPlane = weight->size[0];
    THNN_CHECK_DIM_SIZE(input, ndim, dimf, nInputPlane);
  }

  const int64_t inputWidth   = input->size[dimw];
  const int64_t inputHeight  = input->size[dimh];
  const int64_t inputDepth   = input->size[dimd];
  const int64_t outputDepth  = (inputDepth - 1) * dT - 2*pT + (dilationT * (kT - 1) + 1) + aT;
  const int64_t outputHeight = (inputHeight - 1) * dH - 2*pH + (dilationH * (kH - 1) + 1) + aH;
  const int64_t outputWidth  = (inputWidth - 1) * dW - 2*pW + (dilationW * (kW - 1) + 1) + aW;

  if (outputDepth < 1 || outputWidth < 1 || outputHeight < 1) {
    THError("Given input size per channel: (%ld x %ld x %ld). "
      "Calculated output size per channel: (%ld x %ld x %ld). Output size is too small",
      inputDepth, inputHeight, inputWidth, outputDepth, outputHeight, outputWidth);
  }

  if (gradOutput != NULL) {
    if (weight != NULL) {
      const int64_t nOutputPlane = weight->size[1];
      THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    } else if (bias != NULL) {
      const int64_t nOutputPlane = bias->size[0];
      THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    }
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimd, outputDepth);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
}

void THNN_(VolumetricFullDilatedConvolution_updateOutput)(
  THNNState *state,
  THTensor *input,          // 4D or 5D (batch) tensor
  THTensor *output,
  THTensor *weight,         // weight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
  THTensor *bias,
  THTensor *finput,         // internal columns buffer
  THTensor *fgradInput,     // internal ones buffer
  int kT, int kW, int kH,   // kernel size
  int dT, int dW, int dH,   // stride of the convolution
  int pT, int pW, int pH,   // padding
  int dilationT, int dilationW, int dilationH,
  int aT, int aW, int aH)   // extra output adjustment
{
  THTensor *columns = finput;
  THTensor *ones    = fgradInput;

  THNN_(VolumetricFullDilatedConvolution_shapeCheck)(
        input, NULL, weight, bias, kT, kW, kH,
        dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH, 0);

  const int nInputPlane  = (int)weight->size[0];
  const int nOutputPlane = (int)weight->size[1];

  input = THTensor_(newContiguous)(input);
  weight = THTensor_(newContiguous)(weight);
  bias = bias ? THTensor_(newContiguous)(bias) : bias;
  int is_batch = 1;
  if (input->nDimension == 4)
  {
    // Force batch
    is_batch = 0;
    THTensor_(resize5d)(input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
  }

  const int64_t inputWidth   = input->size[4];
  const int64_t inputHeight  = input->size[3];
  const int64_t inputDepth   = input->size[2];
  const int64_t outputDepth  = (inputDepth - 1) * dT - 2*pT + (dilationT * (kT - 1) + 1) + aT;
  const int64_t outputHeight = (inputHeight - 1) * dH - 2*pH + (dilationH * (kH - 1) + 1) + aH;
  const int64_t outputWidth  = (inputWidth - 1) * dW - 2*pW + (dilationW * (kW - 1) + 1) + aW;

  // Batch size + input planes
  const int64_t batchSize = input->size[0];

  // Resize output
  THTensor_(resize5d)(output, batchSize, nOutputPlane, outputDepth, outputHeight, outputWidth);

  // Resize temporary columns
  THTensor_(resize2d)(columns, nOutputPlane*kW*kH*kT, inputDepth*inputHeight*inputWidth);
  THTensor_(zero)(columns);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 3 || ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth)
  {
    // Resize plane and fill with ones...
    THTensor_(resize3d)(ones, outputDepth, outputHeight, outputWidth);
    THTensor_(fill)(ones, 1);
  }

  // Helpers
  THTensor *input_n = THTensor_(new)();
  THTensor *output_n = THTensor_(new)();

  int elt;
  // For each elt in batch, do:
  for (elt = 0; elt < batchSize; ++elt)
  {
    // Matrix mulitply per output:
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(output_n, output, 0, elt);

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    const int64_t m = weight->size[1] * weight->size[2] * weight->size[3] * weight->size[4];
    const int64_t n = columns->size[1];
    const int64_t k = weight->size[0];

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
    THNN_(col2vol)(
      THTensor_(data)(columns),
      nOutputPlane, outputDepth, outputHeight, outputWidth,
      inputDepth, inputHeight, inputWidth,
      kT, kH, kW,
      pT, pH, pW,
      dT, dH, dW,
      dilationT,  dilationH,  dilationW,
      THTensor_(data)(output_n)
    );

    // Do Bias after:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    const int64_t m_ = nOutputPlane;
    const int64_t n_ = outputDepth * outputHeight * outputWidth;
    const int64_t k_ = 1;

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
  if (is_batch == 0)
  {
    THTensor_(resize4d)(output, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THTensor_(resize4d)(input, nInputPlane, inputDepth, inputHeight, inputWidth);
  }

  THTensor_(free)(input);
  THTensor_(free)(weight);
  if (bias) THTensor_(free)(bias);
}

void THNN_(VolumetricFullDilatedConvolution_updateGradInput)(
  THNNState *state,
  THTensor *input,
  THTensor *gradOutput,
  THTensor *gradInput,
  THTensor *weight,
  THTensor *finput,
  THTensor *fgradInput,     // only used by cuda impl
  int kT, int kW, int kH,   // kernel size
  int dT, int dW, int dH,   // stride
  int pT, int pW, int pH,   // padding
  int dilationT, int dilationW, int dilationH,
  int aT, int aW, int aH)   // extra output adjustment
{
  THTensor *gradColumns = finput;

  // number of input & output planes and kernel size is indirectly defined by the weight tensor
  THNN_(VolumetricFullDilatedConvolution_shapeCheck)(
        input, gradOutput, weight, NULL, kT, kW, kH,
        dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH, 0);

  const int nInputPlane  = (int)weight->size[0];
  const int nOutputPlane = (int)weight->size[1];

  input = THTensor_(newContiguous)(input);
  weight = THTensor_(newContiguous)(weight);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  int is_batch = 1;
  if (input->nDimension == 4)
  {
    // Force batch
    is_batch = 0;
    THTensor_(resize5d)(input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THTensor_(resize5d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  const int64_t inputWidth   = input->size[4];
  const int64_t inputHeight  = input->size[3];
  const int64_t inputDepth   = input->size[2];
  const int64_t outputDepth  = (inputDepth - 1) * dT - 2*pT + (dilationT * (kT - 1) + 1) + aT;
  const int64_t outputHeight = (inputHeight - 1) * dH - 2*pH + (dilationH * (kH - 1) + 1) + aH;
  const int64_t outputWidth  = (inputWidth - 1) * dW - 2*pW + (dilationW * (kW - 1) + 1) + aW;

  // Batch size + input planes
  const int64_t batchSize = input->size[0];

  // Resize output
  THTensor_(resize5d)(gradInput, batchSize, nInputPlane, inputDepth, inputHeight, inputWidth);
  THTensor_(zero)(gradInput);

  // Resize temporary columns
  THTensor_(resize2d)(gradColumns, nOutputPlane*kW*kH*kT, inputDepth*inputHeight*inputWidth);

  // Helpers
  THTensor *gradInput_n = THTensor_(new)();
  THTensor *gradOutput_n = THTensor_(new)();

  int elt;
  // For each elt in batch, do:
  for (elt = 0; elt < batchSize; ++elt)
  {
    // Matrix mulitply per sample:
    THTensor_(select)(gradInput_n, gradInput, 0, elt);
    THTensor_(select)(gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    THNN_(vol2col)(
      THTensor_(data)(gradOutput_n),
      nOutputPlane, outputDepth, outputHeight, outputWidth,
      inputDepth, inputHeight, inputWidth,
      kT, kH, kW,
      pT, pH, pW,
      dT, dH, dW,
      dilationT,  dilationH,  dilationW,
      THTensor_(data)(gradColumns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    const int64_t m = weight->size[0];
    const int64_t n = gradColumns->size[1];
    const int64_t k = weight->size[1] * weight->size[2] * weight->size[3] * weight->size[4];

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
  if (is_batch == 0)
  {
    THTensor_(resize4d)(gradOutput, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THTensor_(resize4d)(input, nInputPlane, inputDepth, inputHeight, inputWidth);
    THTensor_(resize4d)(gradInput, nInputPlane, inputDepth, inputHeight, inputWidth);
  }

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
  THTensor_(free)(weight);
}

void THNN_(VolumetricFullDilatedConvolution_accGradParameters)(
  THNNState *state,
  THTensor *input,
  THTensor *gradOutput,
  THTensor *gradWeight,
  THTensor *gradBias,
  THTensor *finput,
  THTensor *fgradInput,
  int kT, int kW, int kH,   // kernel size
  int dT, int dW, int dH,   // stride
  int pT, int pW, int pH,   // padding
  int dilationT, int dilationW, int dilationH,
  int aT, int aW, int aH,   // extra output adjustment
  accreal scale_)
{
  real scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  // number of input & output planes and kernel size is indirectly defined by the gradWeight tensor
  THNN_(VolumetricFullDilatedConvolution_shapeCheck)(
        input, gradOutput, gradWeight, gradBias, kT, kW, kH,
        dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH, 1);

  int nOutputPlane;
  if (gradWeight) {
    nOutputPlane = THTensor_(size)(gradWeight, 1);
  } else if (gradBias) {
    nOutputPlane = THTensor_(size)(gradBias, 0);
  } else {
    return;
  }

  THTensor *columns = finput;
  THTensor *ones = fgradInput;

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  if (gradWeight) {
    THArgCheck(THTensor_(isContiguous)(gradWeight), 4, "gradWeight needs to be contiguous");
  }
  if (gradBias) {
    THArgCheck(THTensor_(isContiguous)(gradBias), 5, "gradBias needs to be contiguous");
    THArgCheck(THTensor_(isContiguous)(ones), 7, "ones needs to be contiguous");
  }

  int is_batch = 1;
  if (input->nDimension == 4)
  {
    // Force batch
    is_batch = 0;
    THTensor_(resize5d)(input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THTensor_(resize5d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  const int64_t inputWidth   = input->size[4];
  const int64_t inputHeight  = input->size[3];
  const int64_t inputDepth   = input->size[2];
  const int64_t outputDepth  = (inputDepth - 1) * dT - 2*pT + (dilationT * (kT - 1) + 1) + aT;
  const int64_t outputHeight = (inputHeight - 1) * dH - 2*pH + (dilationH * (kH - 1) + 1) + aH;
  const int64_t outputWidth  = (inputWidth - 1) * dW - 2*pW + (dilationW * (kW - 1) + 1) + aW;

  // Batch size + input planes
  const int64_t batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 3 || ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth)
  {
    // Resize plane and fill with ones...
    THTensor_(resize3d)(ones, outputDepth, outputHeight, outputWidth);
    THTensor_(fill)(ones, 1);
  }

  // Resize temporary columns
  THTensor_(resize2d)(columns, nOutputPlane*kW*kH*kT, inputDepth*inputHeight*inputWidth);

  // Helpers
  THTensor *input_n = THTensor_(new)();
  THTensor *gradOutput_n = THTensor_(new)();

  int elt;
  // For each elt in batch, do:
  for (elt = 0; elt < batchSize; ++elt)
  {
    // Matrix mulitply per output:
    THTensor_(select)(gradOutput_n, gradOutput, 0, elt);

    // Do Weight:
    if (gradWeight) {
      // Matrix mulitply per output:
      THTensor_(select)(input_n, input, 0, elt);

      // Extract columns:
      THNN_(vol2col)(
        THTensor_(data)(gradOutput_n), nOutputPlane,
        outputDepth, outputHeight, outputWidth,
        inputDepth, inputHeight, inputWidth,
        kT, kH, kW,
        pT, pH, pW,
        dT, dH, dW,
        dilationT,  dilationH,  dilationW,
        THTensor_(data)(columns)
      );

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      const int64_t n = columns->size[0];   // nOutputPlane * kt * kh * kw
      const int64_t m = input_n->size[0];   // nInputPlane
      const int64_t k = columns->size[1];   // inputHeight * inputWidth

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
      const int64_t m_ = nOutputPlane;
      const int64_t k_ = outputDepth * outputHeight * outputWidth;

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
  if (is_batch == 0)
  {
    THTensor_(resize4d)(gradOutput, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THTensor_(resize4d)(input, input->size[1], inputDepth, inputHeight, inputWidth);
  }

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
}

#endif
