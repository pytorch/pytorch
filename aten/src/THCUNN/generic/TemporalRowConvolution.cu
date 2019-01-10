#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/TemporalRowConvolution.cu"
#else

static inline void THNN_(TemporalRowConvolution_shapeCheck)(
    THCState *state, THCTensor *input, THCTensor *gradOutput, THCTensor *weight,
    THCTensor *bias, int kW, int dW, int padW) {

  THArgCheck(kW > 0, 5,
             "kernel size should be greater than zero, but got kW: %d", kW);
  THArgCheck(dW > 0, 6, "stride should be greater than zero, but got dW: %d",
             dW);
  THCUNN_argCheck(state, weight->nDimension == 2 || weight->nDimension == 3, 3,
                  weight, "2D or 3D weight tensor expected, but got: %s");

  if (bias != NULL) {
    THCUNN_check_dim_size(state, bias, 1, 0, weight->size[0]);
  }

  int ndim = input->nDimension;
  int dimF = 0; // feature dimension
  int dimS = 1; // sequence dimension

  if (ndim == 3) {
    ++dimF;
    ++dimS;
  }

  THCUNN_argCheck(state, ndim == 2 || ndim == 3, 1, input,
                  "2D or 3D (batch mode) input tensor expected, but got :%s");

  int64_t inputFrameSize = weight->size[0];
  int64_t nInputFrame = input->size[dimS];
  int64_t nOutputFrame = (nInputFrame + 2 * padW - kW) / dW + 1;

  if (nOutputFrame < 1) {
    THError("Given input size: (%d x %d). "
            "Calculated output size: (%d x %d). Output size is too small",
            inputFrameSize, nInputFrame, inputFrameSize, nOutputFrame);
  }

  THCUNN_check_dim_size(state, input, ndim, dimF, inputFrameSize);

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, ndim, dimF, inputFrameSize);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimS, nOutputFrame);
  }
}

void THNN_(TemporalRowConvolution_updateOutput)(
    THCState *state, THCTensor *input, THCTensor *output, THCTensor *weight,
    THCTensor *bias, THCTensor *finput, THCTensor *fgradInput, int kW, int dW,
    int padW, bool featFirst) {

  // aliases
  THCTensor *columns = finput;
  THCTensor *ones = fgradInput;

  // assert same GPU
  THCUNN_assertSameGPU(state, 5, input, output, weight, columns, ones);
  if (bias != NULL) {
    THCUNN_assertSameGPU(state, 2, weight, bias);
  }

  THArgCheck(THCTensor_(isContiguous)(state, weight), 4, "weight must be contiguous");
  THArgCheck(!bias || THCTensor_(isContiguous)(state, bias), 5, "bias must be contiguous");

  // reshape weight if necessary
  int ndim = input->nDimension;

  THCTensor *tinput;

  if (!featFirst) {
    tinput = THCTensor_(newTranspose)(state, input, ndim - 1, ndim - 2);
    input = THCTensor_(newContiguous)(state, tinput);
  } else {
    input = THCTensor_(newContiguous)(state, input);
  }

  THNN_(TemporalRowConvolution_shapeCheck)
  (state, input, NULL, weight, bias, kW, dW, padW);

  int batch = 1;
  if (ndim == 2) {
    // Force batch
    batch = 0;
    THCTensor_(resize3d)(state, input, 1, input->size[0], input->size[1]);
  }

  // Params:
  int64_t inputFrameSize = weight->size[0];
  int64_t nInputFrame = input->size[2];
  int64_t nOutputFrame = (nInputFrame + 2 * padW - kW) / dW + 1;

  // Batch size
  int64_t batchSize = input->size[0];

  // Resize output
  THCTensor_(resize3d)(state, output, batchSize, inputFrameSize, nOutputFrame);

  // Augment the input
  THCTensor_(resize3d)(state, columns, inputFrameSize, kW, nOutputFrame);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever
  // gets increased and always contains ones.
  if (ones->nDimension != 2 || ones->size[0] * ones->size[1] < nOutputFrame) {
    // Resize plane and fill with ones...
    THCTensor_(resize2d)(state, ones, 1, nOutputFrame);
    THCTensor_(fill)(state, ones, ScalarConvert<int, real>::to(1));
  }

  // Helpers
  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *output_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; ++elt) {
    // Matrix multiply per output:
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);

    // Do bias first:
    // m_, n_, k_ are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    int64_t m_ = inputFrameSize;
    int64_t n_ = nOutputFrame;
    int64_t k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm asummes
    // column-major matrices)
    if (bias != NULL) {
#ifdef THC_REAL_IS_FLOAT
      THCudaBlas_Sgemm(
#elif defined(THC_REAL_IS_HALF)
      THCudaBlas_Hgemm(
#elif defined(THC_REAL_IS_DOUBLE)
      THCudaBlas_Dgemm(
#endif
          state, 't', 'n', n_, m_, k_, ScalarConvert<int, real>::to(1),
          THCTensor_(data)(state, ones), k_, THCTensor_(data)(state, bias), k_,
          ScalarConvert<int, real>::to(0), THCTensor_(data)(state, output_n),
          n_);
    } else {
      THCTensor_(zero)(state, output_n);
    }

    // Extract columns:
    row2col(THCState_getCurrentStream(state), THCTensor_(data)(state, input_n),
            inputFrameSize, nInputFrame, kW, padW, dW, 1,
            THCTensor_(data)(state, columns));

    THCTensor *output3d = THCTensor_(newWithStorage3d)(
        state, output_n->storage, output_n->storageOffset, inputFrameSize, -1,
        1, -1, nOutputFrame, -1);

    // weight:    inputFrameSize x 1 x kW
    // columns:   inputFrameSize x kW x nOutputFrame
    THCTensor_(baddbmm)(state, output3d, ScalarConvert<int, real>::to(1),
                        output3d, ScalarConvert<int, real>::to(1), weight,
                        columns);
    // output3d:  inputFrameSize x 1 x nOutputFrame

    THCTensor_(free)(state, output3d);
  }

  // Free
  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);

  // Resize output
  if (batch == 0) {
    THCTensor_(resize2d)(state, output, inputFrameSize, nOutputFrame);
    THCTensor_(resize2d)(state, input, inputFrameSize, nInputFrame);
  }

  if (!featFirst) {
    THCTensor_(transpose)(state, output, output, ndim - 1, ndim - 2);
    THCTensor_(free)(state, tinput);
  }

  THCTensor_(free)(state, input);
}

void THNN_(TemporalRowConvolution_updateGradInput)(
    THCState *state, THCTensor *input, THCTensor *gradOutput,
    THCTensor *gradInput, THCTensor *weight, THCTensor *finput,
    THCTensor *fgradInput, int kW, int dW, int padW, bool featFirst) {

  // aliases
  THCTensor *gradColumns = finput;

  THCUNN_assertSameGPU(state, 5, input, gradOutput, weight, gradColumns,
                       gradInput);

  THArgCheck(THCTensor_(isContiguous)(state, weight), 4, "weight must be contiguous");

  int ndim = input->nDimension;

  THCTensor *tinput, *tgradOutput;

  if (!featFirst) {
    tinput = THCTensor_(newTranspose)(state, input, ndim - 1, ndim - 2);
    tgradOutput =
        THCTensor_(newTranspose)(state, gradOutput, ndim - 1, ndim - 2);
    input = THCTensor_(newContiguous)(state, tinput);
    gradOutput = THCTensor_(newContiguous)(state, tgradOutput);

  } else {
    input = THCTensor_(newContiguous)(state, input);
    gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  }

  THNN_(TemporalRowConvolution_shapeCheck)
  (state, input, gradOutput, weight, NULL, kW, dW, padW);

  int batch = 1;
  if (ndim == 2) {
    // Force batch
    batch = 0;
    THCTensor_(resize3d)(state, input, 1, input->size[0], input->size[1]);
    THCTensor_(resize3d)(state, gradOutput, 1, gradOutput->size[0],
                         gradOutput->size[1]);
  }

  // Params:
  int64_t inputFrameSize = weight->size[0];
  int64_t nInputFrame = input->size[2];
  int64_t nOutputFrame = gradOutput->size[2];

  // Batch size
  int64_t batchSize = input->size[0];

  // Resize output
  THCTensor_(resize3d)(state, gradInput, batchSize, inputFrameSize,
                       nInputFrame);

  // Resize temporary columns
  THCTensor_(resize3d)(state, gradColumns, inputFrameSize, kW, nOutputFrame);

  // Helpers
  THCTensor *gradInput_n = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  THCTensor *tweight = THCTensor_(new)(state);
  THCTensor_(transpose)(state, tweight, weight, 1, 2);

  for (int elt = 0; elt < batchSize; ++elt) {
    // Matrix multiply per sample:
    THCTensor_(select)(state, gradInput_n, gradInput, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    THCTensor *gradOutput3d = THCTensor_(newWithStorage3d)(
        state, gradOutput_n->storage, gradOutput_n->storageOffset,
        inputFrameSize, -1, 1, -1, nOutputFrame, -1);

    // weight:          inputFrameSize x kW x 1
    // gradOutput3d:    inputFrameSize x 1 x nOutputFrame
    THCTensor_(baddbmm)(state, gradColumns, ScalarConvert<int, real>::to(0),
                        gradColumns, ScalarConvert<int, real>::to(1), tweight,
                        gradOutput3d);
    // gradColumns:     inputFrameSize x kW x nOutputFrame

    // Unpack columns back into input:
    col2row<real, accreal>(THCState_getCurrentStream(state),
                           THCTensor_(data)(state, gradColumns), inputFrameSize,
                           nInputFrame, kW, padW, dW, 1,
                           THCTensor_(data)(state, gradInput_n));

    THCTensor_(free)(state, gradOutput3d);
  }

  // Free
  THCTensor_(free)(state, gradInput_n);
  THCTensor_(free)(state, gradOutput_n);

  // Resize output
  if (batch == 0) {
    THCTensor_(resize2d)(state, gradOutput, inputFrameSize, nOutputFrame);
    THCTensor_(resize2d)(state, input, inputFrameSize, nInputFrame);
    THCTensor_(resize2d)(state, gradInput, inputFrameSize, nInputFrame);
  }

  THCTensor_(free)(state, tweight);

  if (!featFirst) {
    THCTensor_(transpose)(state, gradInput, gradInput, ndim - 1, ndim - 2);
    THCTensor_(free)(state, tinput);
    THCTensor_(free)(state, tgradOutput);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

void THNN_(TemporalRowConvolution_accGradParameters)(
    THCState *state, THCTensor *input, THCTensor *gradOutput,
    THCTensor *gradWeight, THCTensor *gradBias, THCTensor *finput,
    THCTensor *fgradInput, int kW, int dW, int padW, bool featFirst,
    accreal scale_) {

  real scale = ScalarConvert<accreal, real>::to(scale_);
  // Aliases
  THCTensor *columns = finput;
  THCTensor *ones = fgradInput;

  THCUNN_assertSameGPU(state, 5, input, gradOutput, gradWeight, columns, ones);
  if (gradBias != NULL) {
    THCUNN_assertSameGPU(state, 2, gradWeight, gradBias);
  }

  int ndim = input->nDimension;

  THCTensor *tinput, *tgradOutput;

  if (!featFirst) {
    tinput = THCTensor_(newTranspose)(state, input, ndim - 1, ndim - 2);
    tgradOutput =
        THCTensor_(newTranspose)(state, gradOutput, ndim - 1, ndim - 2);
    input = THCTensor_(newContiguous)(state, tinput);
    gradOutput = THCTensor_(newContiguous)(state, tgradOutput);
  } else {
    input = THCTensor_(newContiguous)(state, input);
    gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  }

  THNN_(TemporalRowConvolution_shapeCheck)
  (state, input, gradOutput, gradWeight, gradBias, kW, dW, padW);

  int batch = 1;
  if (ndim == 2) {
    // Force batch
    batch = 0;
    THCTensor_(resize3d)(state, input, 1, input->size[0], input->size[1]);
    THCTensor_(resize3d)(state, gradOutput, 1, gradOutput->size[0],
                         gradOutput->size[1]);
  }

  // Params:
  int64_t inputFrameSize = gradWeight->size[0];
  int64_t nInputFrame = input->size[2];
  int64_t nOutputFrame = gradOutput->size[2];

  // Batch size
  int64_t batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 || ones->size[0] * ones->size[1] < nOutputFrame) {
    // Resize plane and fill with ones...
    THCTensor_(resize2d)(state, ones, 1, nOutputFrame);
    THCTensor_(fill)(state, ones, ScalarConvert<int, real>::to(1));
  }

  // // Resize temporary columns
  THCTensor_(resize3d)(state, columns, inputFrameSize, kW, nOutputFrame);

  // Helpers
  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; ++elt) {
    // Matrix multiply per output
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    THCTensor *gradOutput3d = THCTensor_(newWithStorage3d)(
        state, gradOutput_n->storage, gradOutput_n->storageOffset,
        inputFrameSize, -1, 1, -1, nOutputFrame, -1);

    // Extract columns
    row2col(THCState_getCurrentStream(state), THCTensor_(data)(state, input_n),
            inputFrameSize, nInputFrame, kW, padW, dW, 1,
            THCTensor_(data)(state, columns));

    THCTensor *tcolumns = THCTensor_(new)(state);
    THCTensor_(transpose)(state, tcolumns, columns, 1, 2);

    // gradOutput3d:  inputFrameSize x 1 x nOutputFrame
    // columns:       inputFrameSize x nOutputFrame x kW
    THCTensor_(baddbmm)(state, gradWeight, ScalarConvert<int, real>::to(1),
                        gradWeight, scale, gradOutput3d, tcolumns);
    // gradWeight:    inputFrameSize x 1 x kW

    THCTensor_(free)(state, tcolumns);
    THCTensor_(free)(state, gradOutput3d);

    if (gradBias != NULL) {
      int64_t m_ = inputFrameSize;
      int64_t k_ = nOutputFrame;
#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
#ifdef THC_REAL_IS_FLOAT
      THCudaBlas_Sgemv(
#elif defined(THC_REAL_IS_DOUBLE)
      THCudaBlas_Dgemv(
#endif
          state, 't', k_, m_, scale, THCTensor_(data)(state, gradOutput_n), k_,
          THCTensor_(data)(state, ones), 1, ScalarConvert<int, real>::to(1),
          THCTensor_(data)(state, gradBias), 1);
#endif
#ifdef THC_REAL_IS_HALF // half not supported due to baddbmm
      THCudaBlas_Hgemm(state, 't', 'n', m_, 1, k_, scale,
                       THCTensor_(data)(state, gradOutput_n), k_,
                       THCTensor_(data)(state, ones), k_,
                       ScalarConvert<int, real>::to(1),
                       THCTensor_(data)(state, gradBias), m_);
#endif
    }
  }

  // Free
  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, gradOutput_n);

  // Resize
  if (batch == 0) {
    THCTensor_(resize2d)(state, gradOutput, inputFrameSize, nOutputFrame);
    THCTensor_(resize2d)(state, input, inputFrameSize, nInputFrame);
  }

  if (!featFirst) {
    THCTensor_(free)(state, tinput);
    THCTensor_(free)(state, tgradOutput);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

#endif
