#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricFullConvolution.c"
#else


static void nn_(vol2col)(const real* data_vol, const int channels,
    const int depth, const int height, const int width, const int kernel_t, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    real* data_col) {
  int c, t, h, w;
  int depth_col = (depth + 2 * pad_t - kernel_t) / stride_t + 1;
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_t * kernel_h * kernel_w;
  for (c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int t_offset = (c / kernel_w / kernel_h) % kernel_t;
    int c_vol = c / kernel_t / kernel_h / kernel_w;
    for (t = 0; t < depth_col; ++t) {
      for (h = 0; h < height_col; ++h) {
        for (w = 0; w < width_col; ++w) {
          int t_pad = t * stride_t - pad_t + t_offset;
          int h_pad = h * stride_h - pad_h + h_offset;
          int w_pad = w * stride_w - pad_w + w_offset;
          if (t_pad >= 0 && t_pad < depth && h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
            data_col[((c * depth_col + t) * height_col + h) * width_col + w] =
              data_vol[((c_vol * depth + t_pad) * height + h_pad) * width + w_pad];
          else
            data_col[((c * depth_col + t) * height_col + h) * width_col + w] = 0;
        }
      }
    }
  }
}

static void nn_(col2vol)(const real* data_col, const int channels,
    const int depth, const int height, const int width, const int patch_t, const int patch_h, const int patch_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    real* data_vol) {
  int c, t, h, w;
  memset(data_vol, 0, sizeof(real) * depth * height * width * channels);
  int depth_col = (depth + 2 * pad_t - patch_t) / stride_t + 1;
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int channels_col = channels * patch_t * patch_h * patch_w;
  for (c = 0; c < channels_col; ++c) {
    int w_offset = c % patch_w;
    int h_offset = (c / patch_w) % patch_h;
    int t_offset = (c / patch_w / patch_h) % patch_t;
    int c_vol = c / patch_t / patch_h / patch_w;
    for (t = 0; t < depth_col; ++t) {
      for (h = 0; h < height_col; ++h) {
        for (w = 0; w < width_col; ++w) {
          int t_pad = t * stride_t - pad_t + t_offset;
          int h_pad = h * stride_h - pad_h + h_offset;
          int w_pad = w * stride_w - pad_w + w_offset;
          if (t_pad >= 0 && t_pad < depth && h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
            data_vol[((c_vol * depth + t_pad) * height + h_pad) * width + w_pad] +=
              data_col[((c * depth_col + t) * height_col + h) * width_col + w];
        }
      }
    }
  }
}

static int nn_(VolumetricFullConvolution_updateOutput)(lua_State *L) {
  // Input
  THTensor *input = (THTensor*)luaT_checkudata(L, 2, torch_Tensor);

  // Params:
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int dT = luaT_getfieldcheckint(L, 1, "dT");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int kT = luaT_getfieldcheckint(L, 1, "kT");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  int padT = luaT_getfieldcheckint(L, 1, "padT");
  int adjW = luaT_getfieldcheckint(L, 1, "adjW");
  int adjH = luaT_getfieldcheckint(L, 1, "adjH");
  int adjT = luaT_getfieldcheckint(L, 1, "adjT");

  THTensor *weight  = (THTensor*)luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *bias    = (THTensor*)luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *columns = (THTensor*)luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *ones    = (THTensor*)luaT_getfieldcheckudata(L, 1, "fgradInput", torch_Tensor);
  THTensor *output  = (THTensor*)luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5, 2, "4D or 5D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 4) {
    luaL_argcheck(L, input->size[0] == nInputPlane, 2, "input channels and nInputPlane dont match");
    // Force batch
    batch = 0;
    THTensor_(resize5d)(input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
  } else {
    luaL_argcheck(L, input->size[1] == nInputPlane, 2, "input channels and nInputPlane dont match");
  }

  long inputWidth   = input->size[4];
  long inputHeight  = input->size[3];
  long inputDepth = input->size[2];
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + kW + adjW;
  long outputHeight = (inputHeight - 1) * dH - 2*padH + kH + adjH;
  long outputDepth = (inputDepth - 1) * dT - 2*padT + kT + adjT;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THTensor_(resize5d)(output, batchSize, nOutputPlane, outputDepth, outputHeight, outputWidth);

  // Resize temporary columns
  THTensor_(resize2d)(columns, nOutputPlane*kW*kH*kT, inputDepth*inputHeight*inputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 3 || ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THTensor_(resize3d)(ones, outputDepth, outputHeight, outputWidth);
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
    long m = weight->size[1] * weight->size[2] * weight->size[3] * weight->size[4];
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
    nn_(col2vol)(
      THTensor_(data)(columns),
      nOutputPlane, outputDepth, outputHeight, outputWidth, kT, kH, kW, padT, padH, padW, dT, dH, dW,
      THTensor_(data)(output_n)
    );

    // Do Bias after:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long n_ = outputDepth * outputHeight * outputWidth;
    long k_ = 1;

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
  if (batch == 0) {
    THTensor_(resize4d)(output, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THTensor_(resize4d)(input, nInputPlane, inputDepth, inputHeight, inputWidth);
  }

  // return output
  return 1;
}

static int nn_(VolumetricFullConvolution_updateGradInput)(lua_State *L) {
  // Inputs
  THTensor *input = (THTensor *)luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = (THTensor *)luaT_checkudata(L, 3, torch_Tensor);

  // Params
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int dT = luaT_getfieldcheckint(L, 1, "dT");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int kT = luaT_getfieldcheckint(L, 1, "kT");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  int padT = luaT_getfieldcheckint(L, 1, "padT");
  int adjW = luaT_getfieldcheckint(L, 1, "adjW");
  int adjH = luaT_getfieldcheckint(L, 1, "adjH");
  int adjT = luaT_getfieldcheckint(L, 1, "adjT");

  THTensor *weight = (THTensor *)luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradColumns = (THTensor*)luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *gradInput = (THTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5, 2, "4D or 5D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THTensor_(resize5d)(input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THTensor_(resize5d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputWidth   = input->size[4];
  long inputHeight  = input->size[3];
  long inputDepth   = input->size[2];
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + kW + adjW;
  long outputHeight = (inputHeight - 1) * dH - 2*padH + kH + adjH;
  long outputDepth = (inputDepth - 1) * dT - 2*padT + kT + adjT;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THTensor_(resize5d)(gradInput, batchSize, nInputPlane, inputDepth, inputHeight, inputWidth);

  // Resize temporary columns
  THTensor_(resize2d)(gradColumns, nOutputPlane*kW*kH*kT, inputDepth*inputHeight*inputWidth);

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
    nn_(vol2col)(
      THTensor_(data)(gradOutput_n),
      nOutputPlane, outputDepth, outputHeight, outputWidth, kT, kH, kW, padT, padH, padW, dT, dH, dW,
      THTensor_(data)(gradColumns)
    );


    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[0];
    long n = gradColumns->size[1];
    long k = weight->size[1] * weight->size[2] * weight->size[3] * weight->size[4];

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
    THTensor_(resize4d)(gradOutput, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THTensor_(resize4d)(input, nInputPlane, inputDepth, inputHeight, inputWidth);
    THTensor_(resize4d)(gradInput, nInputPlane, inputDepth, inputHeight, inputWidth);
  }

  // Return gradInput
  return 1;
}


static int nn_(VolumetricFullConvolution_accGradParameters)(lua_State *L) {
  // Inputs
  THTensor *input = (THTensor *)luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = (THTensor *)luaT_checkudata(L, 3, torch_Tensor);

  // Params
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int dT = luaT_getfieldcheckint(L, 1, "dT");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int kT = luaT_getfieldcheckint(L, 1, "kT");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  int padT = luaT_getfieldcheckint(L, 1, "padT");
  int adjW = luaT_getfieldcheckint(L, 1, "adjW");
  int adjH = luaT_getfieldcheckint(L, 1, "adjH");
  int adjT = luaT_getfieldcheckint(L, 1, "adjT");
  float scale = luaL_optnumber(L, 4, 1);

  THTensor *gradWeight = (THTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *gradBias = (THTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  THTensor *columns = (THTensor*)luaT_getfieldcheckudata(L, 1, "finput", torch_Tensor);
  THTensor *ones = (THTensor*)luaT_getfieldcheckudata(L, 1, "fgradInput", torch_Tensor);

  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5, 2, "4D or 5D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THTensor_(resize5d)(input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THTensor_(resize5d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputWidth   = input->size[4];
  long inputHeight  = input->size[3];
  long inputDepth  = input->size[2];
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + kW + adjW;
  long outputHeight = (inputHeight - 1) * dH - 2*padH + kH + adjH;
  long outputDepth = (inputDepth - 1) * dT - 2*padT + kT + adjT;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 3 || ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth) {
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
  for (elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    nn_(vol2col)(
      THTensor_(data)(gradOutput_n),
      nOutputPlane, outputDepth, outputHeight, outputWidth, kT, kH, kW, padT, padH, padW, dT, dH, dW,
      THTensor_(data)(columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long n = columns->size[0];   // nOutputPlane * kt * kh * kw
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
    long k_ = outputDepth * outputHeight * outputWidth;

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
  if (batch == 0) {
    THTensor_(resize4d)(gradOutput, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THTensor_(resize4d)(input, nInputPlane, inputDepth, inputHeight, inputWidth);
  }

  // Return nothing
  return 0;
}

static const struct luaL_Reg nn_(VolumetricFullConvolution__) [] = {
  {"VolumetricFullConvolution_updateOutput", nn_(VolumetricFullConvolution_updateOutput)},
  {"VolumetricFullConvolution_updateGradInput", nn_(VolumetricFullConvolution_updateGradInput)},
  {"VolumetricFullConvolution_accGradParameters", nn_(VolumetricFullConvolution_accGradParameters)},
  {NULL, NULL}
};

static void nn_(VolumetricFullConvolution_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(VolumetricFullConvolution__), "nn");
  lua_pop(L,1);
}

#endif
