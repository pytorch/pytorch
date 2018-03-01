#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricGridSamplerBilinear.cu"
#else

static inline void THNN_(VolumetricGridSamplerBilinear_shapeCheck)(
    THCState *state,
    THCTensor *input,
    THCTensor *grid,
    THCTensor *gradOutput) {
  THCUNN_argCheck(state, THCTensor_(nDimension)(state, input) == 5, 2, input,
      "5D input tensor expected but got: %s");
  THCUNN_argCheck(state, THCTensor_(nDimension)(state, grid) == 5, 2, grid,
      "5D grid tensor expected but got: %s");

  int64_t nbatch   = THCTensor_(size)(state, input, 0);
  int64_t channels = THCTensor_(size)(state, input, 1);
  int64_t idepth   = THCTensor_(size)(state, input, 2);
  int64_t iheight   = THCTensor_(size)(state, input, 3);
  int64_t iwidth    = THCTensor_(size)(state, input, 4);
  int64_t odepth   = THCTensor_(size)(state, grid, 1);
  int64_t oheight   = THCTensor_(size)(state, grid, 2);
  int64_t owidth    = THCTensor_(size)(state, grid, 3);

  THCUNN_check_dim_size(state, grid, 5, 0, nbatch);
  THCUNN_check_dim_size(state, grid, 5, 4, 3);

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, 5, 0, nbatch);
    THCUNN_check_dim_size(state, gradOutput, 5, 1, channels);
    THCUNN_check_dim_size(state, gradOutput, 5, 2, odepth);
    THCUNN_check_dim_size(state, gradOutput, 5, 3, oheight);
    THCUNN_check_dim_size(state, gradOutput, 5, 4, owidth);
  }
}

TH_API void THNN_(VolumetricGridSamplerBilinear_updateOutput)(
    THCState *state,
    THCTensor *input,
    THCTensor *grid,
    THCTensor *output,
    int padding_mode) {

  THCUNN_assertSameGPU(state, 3, input, grid, output);
  THNN_(VolumetricGridSamplerBilinear_shapeCheck)(state, input, grid, NULL);
  int64_t N = THCTensor_(size)(state, input, 0);
  int64_t C = THCTensor_(size)(state, input, 1);
  int64_t ID = THCTensor_(size)(state, input, 2);
  int64_t IH = THCTensor_(size)(state, input, 3);
  int64_t IW = THCTensor_(size)(state, input, 4);
  int64_t D = THCTensor_(size)(state,grid, 1);
  int64_t H = THCTensor_(size)(state,grid, 2);
  int64_t W = THCTensor_(size)(state, grid, 3);

  // resize output to the same shape as input
  THCTensor_(resize5d)(state, output, N, C, D, H, W);

  THCDeviceTensor<real, 5> devInput = toDeviceTensor<real, 5>(state, input);
  THCDeviceTensor<real, 5> devGrid = toDeviceTensor<real, 5>(state, grid);
  THCDeviceTensor<real, 5> devOutput = toDeviceTensor<real, 5>(state, output);

  int count = static_cast<int>(N*D*H*W);
  VolumetricGridSamplerBilinear_updateOutput_kernel
    <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      count, devInput, devGrid, devOutput, padding_mode);
  THCudaCheck(cudaGetLastError());
}

TH_API void THNN_(VolumetricGridSamplerBilinear_updateGradInput)(
    THCState *state,
    THCTensor *input, THCTensor *gradInput,
    THCTensor *grid, THCTensor *gradGrid,
    THCTensor *gradOutput,
    int padding_mode) {

  THCUNN_assertSameGPU(state, 5, input, gradInput, grid, gradGrid, gradOutput);
  THNN_(VolumetricGridSamplerBilinear_shapeCheck)(state, input, grid, gradOutput);
  int64_t N = THCTensor_(size)(state, input, 0);
  int64_t C = THCTensor_(size)(state, input, 1);
  int64_t ID = THCTensor_(size)(state, input, 2);
  int64_t IH = THCTensor_(size)(state, input, 3);
  int64_t IW = THCTensor_(size)(state, input, 4);
  int64_t D = THCTensor_(size)(state,grid, 1);
  int64_t H = THCTensor_(size)(state,grid, 2);
  int64_t W = THCTensor_(size)(state, grid, 3);

  THCTensor_(resize5d)(state, gradInput, N, C, ID, IH, IW);
  THCTensor_(resize5d)(state, gradGrid, N, D, H, W, 3);
  THCTensor_(zero)(state, gradInput);
  THCTensor_(zero)(state, gradGrid);

  THCDeviceTensor<real, 5> devInput = toDeviceTensor<real, 5>(state, input);
  THCDeviceTensor<real, 5> devGradInput = toDeviceTensor<real, 5>(state, gradInput);
  THCDeviceTensor<real, 5> devGrid = toDeviceTensor<real, 5>(state, grid);
  THCDeviceTensor<real, 5> devGradGrid = toDeviceTensor<real, 5>(state, gradGrid);
  THCDeviceTensor<real, 5> devGradOutput = toDeviceTensor<real, 5>(state, gradOutput);

  int count = static_cast<int>(N*D*H*W);
  VolumetricGridSamplerBilinear_updateGradInput_kernel
    <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      count, devInput, devGradInput, devGrid, devGradGrid, devGradOutput, padding_mode);
  THCudaCheck(cudaGetLastError());
}

#endif
