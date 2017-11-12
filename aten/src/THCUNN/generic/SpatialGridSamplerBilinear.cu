#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialGridSamplerBilinear.cu"
#else

static inline void THNN_(SpatialGridSamplerBilinear_shapeCheck)(
    THCState *state,
    THCTensor *input,
    THCTensor *grid,
    THCTensor *gradOutput) {
  THCUNN_argCheck(state, THCTensor_(nDimension)(state, input) == 4, 2, input,
      "4D input tensor expected but got: %s");
  THCUNN_argCheck(state, THCTensor_(nDimension)(state, grid) == 4, 2, grid,
      "4D grid tensor expected but got: %s");

  int64_t nbatch   = THCTensor_(size)(state, input, 0);
  int64_t channels = THCTensor_(size)(state, input, 1);
  int64_t iheight   = THCTensor_(size)(state, input, 2);
  int64_t iwidth    = THCTensor_(size)(state, input, 3);
  int64_t oheight   = THCTensor_(size)(state, grid, 1);
  int64_t owidth    = THCTensor_(size)(state, grid, 2);

  THCUNN_check_dim_size(state, grid, 4, 0, nbatch);
  THCUNN_check_dim_size(state, grid, 4, 3, 2);

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, 4, 0, nbatch);
    THCUNN_check_dim_size(state, gradOutput, 4, 1, channels);
    THCUNN_check_dim_size(state, gradOutput, 4, 2, oheight);
    THCUNN_check_dim_size(state, gradOutput, 4, 3, owidth);
  }
}

TH_API void THNN_(SpatialGridSamplerBilinear_updateOutput)(
    THCState *state,
    THCTensor *input,
    THCTensor *grid,
    THCTensor *output,
    int padding_mode) {

  THCUNN_assertSameGPU(state, 3, input, grid, output);
  THNN_(SpatialGridSamplerBilinear_shapeCheck)(state, input, grid, NULL);
  int64_t N = THCTensor_(size)(state, input, 0);
  int64_t C = THCTensor_(size)(state, input, 1);
  int64_t IH = THCTensor_(size)(state, input, 2);
  int64_t IW = THCTensor_(size)(state, input, 3);
  int64_t H = THCTensor_(size)(state,grid, 1);
  int64_t W = THCTensor_(size)(state, grid, 2);

  // resize output to the same shape as input
  THCTensor_(resize4d)(state, output, N, C, H, W);

  THCDeviceTensor<real, 4> devInput = toDeviceTensor<real, 4>(state, input);
  THCDeviceTensor<real, 4> devGrid = toDeviceTensor<real, 4>(state, grid);
  THCDeviceTensor<real, 4> devOutput = toDeviceTensor<real, 4>(state, output);

  int count = static_cast<int>(N*H*W);
  SpatialGridSamplerBilinear_updateOutput_kernel
    <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      count, devInput, devGrid, devOutput, padding_mode);
  THCudaCheck(cudaGetLastError());
}

TH_API void THNN_(SpatialGridSamplerBilinear_updateGradInput)(
    THCState *state,
    THCTensor *input, THCTensor *gradInput,
    THCTensor *grid, THCTensor *gradGrid,
    THCTensor *gradOutput,
    int padding_mode) {

  THCUNN_assertSameGPU(state, 5, input, gradInput, grid, gradGrid, gradOutput);
  THNN_(SpatialGridSamplerBilinear_shapeCheck)(state, input, grid, gradOutput);
  int64_t N = THCTensor_(size)(state, input, 0);
  int64_t C = THCTensor_(size)(state, input, 1);
  int64_t IH = THCTensor_(size)(state, input, 2);
  int64_t IW = THCTensor_(size)(state, input, 3);
  int64_t H = THCTensor_(size)(state, grid, 1);
  int64_t W = THCTensor_(size)(state, grid, 2);

  THCTensor_(resize4d)(state, gradInput, N, C, IH, IW);
  THCTensor_(resize4d)(state, gradGrid, N, H, W, 2);
  THCTensor_(zero)(state, gradInput);
  THCTensor_(zero)(state, gradGrid);

  THCDeviceTensor<real, 4> devInput = toDeviceTensor<real, 4>(state, input);
  THCDeviceTensor<real, 4> devGradInput = toDeviceTensor<real, 4>(state, gradInput);
  THCDeviceTensor<real, 4> devGrid = toDeviceTensor<real, 4>(state, grid);
  THCDeviceTensor<real, 4> devGradGrid = toDeviceTensor<real, 4>(state, gradGrid);
  THCDeviceTensor<real, 4> devGradOutput = toDeviceTensor<real, 4>(state, gradOutput);

  int count = static_cast<int>(N*H*W);
  SpatialGridSamplerBilinear_updateGradInput_kernel
    <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      count, devInput, devGradInput, devGrid, devGradGrid, devGradOutput, padding_mode);
  THCudaCheck(cudaGetLastError());
}

#endif
