#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricFullConvolution.c"
#else

static void THNN_(vol2col)(
  const real *data_vol, const int channels,
  const int depth, const int height, const int width,
  const int kT, const int kH, const int kW,
  const int pT, const int pH, const int pW,
  const int dT, const int dH, const int dW,
  const int dilationT, const int dilationH, const int dilationW,
  real *data_col)
{
  int c, t, h, w;
  int depth_col  = (depth  + 2 * pT - (dilationT * (kT - 1) + 1)) / dT + 1;
  int height_col = (height + 2 * pH - (dilationH * (kH - 1) + 1)) / dH + 1;
  int width_col  = (width  + 2 * pW - (dilationW * (kW - 1) + 1)) / dW + 1;
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
  const int kT, const int kH, const int kW,
  const int pT, const int pH, const int pW,
  const int dT, const int dH, const int dW,
  const int dilationT, const int dilationH, const int dilationW,
  real* data_vol)
{
  int c, t, h, w;
  memset(data_vol, 0, sizeof(real) * depth * height * width * channels);
  int depth_col  = (depth  + 2 * pT - (dilationT * (kT - 1) + 1)) / dT + 1;
  int height_col = (height + 2 * pH - (dilationH * (kH - 1) + 1)) / dH + 1;
  int width_col  = (width  + 2 * pW - (dilationW * (kW - 1) + 1)) / dW + 1;
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

static inline void THNN_(VolumetricFullConvolution_shapeCheck)(
                         THTensor *input, THTensor *gradOutput,
                         THTensor *weight, THTensor *bias,
                         int dT, int dW, int dH, int pT, int pW, int pH,
                         int aT, int aW, int aH) {
  THNN_ARGCHECK(input->nDimension == 4 || input->nDimension == 5, 2, input,
                "4D or 5D (batch mode) tensor expected for input, but got: %s");
  // number of input & output planes and kernel size is indirectly defined by the weight tensor
  THNN_ARGCHECK(weight->nDimension == 5, 4, weight,
                "5D (nOutputPlane x nInputPlane x kT x kH x kW) tensor "
                "expected for weight, but got: %s");
  THArgCheck(dT > 0 && dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dT: %d dH: %d dW: %d", dT, dH, dW);
  THArgCheck(aT < dT && aW < dW && aH < dH, 15,
             "output adjustment must be smaller than stride, but got "
             "adjT: %d adjH: %d adjW: %d dT: %d dH: %d dW: %d",
             aT, aH, aW, dT, dH, dW);

  int ndim = input->nDimension;
  const int nInputPlane  = (int)weight->size[0];
  const int nOutputPlane = (int)weight->size[1];
  const int kT           = (int)weight->size[2];
  const int kH           = (int)weight->size[3];
  const int kW           = (int)weight->size[4];

  if (bias != NULL) {
    THNN_CHECK_DIM_SIZE(bias, 1, 0, weight->size[1]);
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

  const long inputWidth   = input->size[dimw];
  const long inputHeight  = input->size[dimh];
  const long inputDepth   = input->size[dimd];
  const long outputWidth  = (inputWidth  - 1) * dW - 2*pW + kW + aW;
  const long outputHeight = (inputHeight - 1) * dH - 2*pH + kH + aH;
  const long outputDepth  = (inputDepth  - 1) * dT - 2*pT + kT + aT;

  if (outputDepth < 1 || outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%dx%dx%dx%d). Calculated output size: (%dx%dx%dx%d). Output size is too small",
            nInputPlane,inputDepth,inputHeight,inputWidth,nOutputPlane,outputDepth,outputHeight,outputWidth);

  THNN_CHECK_DIM_SIZE(input, ndim, dimf, nInputPlane);
  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimd, outputDepth);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
}

void THNN_(VolumetricFullConvolution_updateOutput)(
  THNNState *state,
  THTensor *input,          // 4D or 5D (batch) tensor
  THTensor *output,
  THTensor *weight,         // weight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
  THTensor *bias,
  THTensor *finput,         // internal columns buffer
  THTensor *fgradInput,     // internal ones buffer
  int dT, int dW, int dH,   // stride of the convolution
  int pT, int pW, int pH,   // padding
  int aT, int aW, int aH)   // extra output adjustment
{
  THTensor *columns = finput;
  THTensor *ones    = fgradInput;

  THNN_(VolumetricFullConvolution_shapeCheck)(
        input, NULL, weight, bias,
        dT, dW, dH, pT, pW, pH, aT, aW, aH);

  const int nInputPlane  = (int)weight->size[0];
  const int nOutputPlane = (int)weight->size[1];
  const int kT           = (int)weight->size[2];
  const int kH           = (int)weight->size[3];
  const int kW           = (int)weight->size[4];

  input = THTensor_(newContiguous)(input);
  int batch = 1;
  if (input->nDimension == 4)
  {
    // Force batch
    batch = 0;
    THTensor_(resize5d)(input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
  }

  const long inputWidth   = input->size[4];
  const long inputHeight  = input->size[3];
  const long inputDepth   = input->size[2];
  const long outputWidth  = (inputWidth  - 1) * dW - 2*pW + kW + aW;
  const long outputHeight = (inputHeight - 1) * dH - 2*pH + kH + aH;
  const long outputDepth  = (inputDepth  - 1) * dT - 2*pT + kT + aT;

  // Batch size + input planes
  const long batchSize = input->size[0];

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
    const long m = weight->size[1] * weight->size[2] * weight->size[3] * weight->size[4];
    const long n = columns->size[1];
    const long k = weight->size[0];

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
      kT, kH, kW,
      pT, pH, pW,
      dT, dH, dW,
       1,  1,  1,
      THTensor_(data)(output_n)
    );

    // Do Bias after:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    const long m_ = nOutputPlane;
    const long n_ = outputDepth * outputHeight * outputWidth;
    const long k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
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

  // Free
  THTensor_(free)(input_n);
  THTensor_(free)(output_n);

  // Resize output
  if (batch == 0)
  {
    THTensor_(resize4d)(output, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THTensor_(resize4d)(input, nInputPlane, inputDepth, inputHeight, inputWidth);
  }

  THTensor_(free)(input);
}

void THNN_(VolumetricFullConvolution_updateGradInput)(
  THNNState *state,
  THTensor *input,
  THTensor *gradOutput,
  THTensor *gradInput,
  THTensor *weight,
  THTensor *finput,
  THTensor *fgradInput,     // only used by cuda impl
  int dT, int dW, int dH,   // stride
  int pT, int pW, int pH,   // padding
  int aT, int aW, int aH)   // extra output adjustment
{
  THTensor *gradColumns = finput;

  // number of input & output planes and kernel size is indirectly defined by the weight tensor
  THNN_(VolumetricFullConvolution_shapeCheck)(
        input, gradOutput, weight, NULL,
        dT, dW, dH, pT, pW, pH, aT, aW, aH);

  const int nInputPlane  = (int)weight->size[0];
  const int nOutputPlane = (int)weight->size[1];
  const int kT           = (int)weight->size[2];
  const int kH           = (int)weight->size[3];
  const int kW           = (int)weight->size[4];

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  int batch = 1;
  if (input->nDimension == 4)
  {
    // Force batch
    batch = 0;
    THTensor_(resize5d)(input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THTensor_(resize5d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  const long inputWidth   = input->size[4];
  const long inputHeight  = input->size[3];
  const long inputDepth   = input->size[2];
  const long outputWidth  = (inputWidth  - 1) * dW - 2*pW + kW + aW;
  const long outputHeight = (inputHeight - 1) * dH - 2*pH + kH + aH;
  const long outputDepth  = (inputDepth  - 1) * dT - 2*pT + kT + aT;

  // Batch size + input planes
  const long batchSize = input->size[0];

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
      kT, kH, kW,
      pT, pH, pW,
      dT, dH, dW,
       1,  1,  1,
      THTensor_(data)(gradColumns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    const long m = weight->size[0];
    const long n = gradColumns->size[1];
    const long k = weight->size[1] * weight->size[2] * weight->size[3] * weight->size[4];

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
  if (batch == 0)
  {
    THTensor_(resize4d)(gradOutput, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THTensor_(resize4d)(input, nInputPlane, inputDepth, inputHeight, inputWidth);
    THTensor_(resize4d)(gradInput, nInputPlane, inputDepth, inputHeight, inputWidth);
  }

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
}

void THNN_(VolumetricFullConvolution_accGradParameters)(
  THNNState *state,
  THTensor *input,
  THTensor *gradOutput,
  THTensor *gradWeight,
  THTensor *gradBias,
  THTensor *finput,
  THTensor *fgradInput,
  int dT, int dW, int dH,   // stride
  int pT, int pW, int pH,   // padding
  int aT, int aW, int aH,   // extra output adjustment
  real scale)
{
  // number of input & output planes and kernel size is indirectly defined by the gradWeight tensor
  THNN_(VolumetricFullConvolution_shapeCheck)(
        input, gradOutput, gradWeight, gradBias,
        dT, dW, dH, pT, pW, pH, aT, aW, aH);

  int nInputPlane  = (int)gradWeight->size[0];
  int nOutputPlane = (int)gradWeight->size[1];
  int kT           = (int)gradWeight->size[2];
  int kH           = (int)gradWeight->size[3];
  int kW           = (int)gradWeight->size[4];

  THTensor *columns = finput;
  THTensor *ones = fgradInput;

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  int batch = 1;
  if (input->nDimension == 4)
  {
    // Force batch
    batch = 0;
    THTensor_(resize5d)(input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THTensor_(resize5d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  const long inputWidth   = input->size[4];
  const long inputHeight  = input->size[3];
  const long inputDepth   = input->size[2];
  const long outputWidth  = (inputWidth  - 1) * dW - 2*pW + kW + aW;
  const long outputHeight = (inputHeight - 1) * dH - 2*pH + kH + aH;
  const long outputDepth  = (inputDepth  - 1) * dT - 2*pT + kT + aT;

  // Batch size + input planes
  const long batchSize = input->size[0];

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
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    THNN_(vol2col)(
      THTensor_(data)(gradOutput_n), nOutputPlane,
      outputDepth, outputHeight, outputWidth,
      kT, kH, kW,
      pT, pH, pW,
      dT, dH, dW,
       1,  1,  1,
      THTensor_(data)(columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    const long n = columns->size[0];   // nOutputPlane * kt * kh * kw
    const long m = input_n->size[0];   // nInputPlane
    const long k = columns->size[1];   // inputHeight * inputWidth

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
    const long m_ = nOutputPlane;
    const long k_ = outputDepth * outputHeight * outputWidth;

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

  // Free
  THTensor_(free)(input_n);
  THTensor_(free)(gradOutput_n);

  // Resize
  if (batch == 0)
  {
    THTensor_(resize4d)(gradOutput, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THTensor_(resize4d)(input, nInputPlane, inputDepth, inputHeight, inputWidth);
  }

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
}

#endif
