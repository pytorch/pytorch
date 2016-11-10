#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/TemporalMaxPooling.cu"
#else

void THNN_(TemporalMaxPooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCIndexTensor *indices,
           int kW, int dW) {

  int dimT = 0; // Temporal dimension
  int dimF = 1; // Feature dimension

  int batch = 1;
  int input_w;
  int input_n;
  int output_w;
  int nthreads;

  real *input_data;
  real *output_data;
  THCIndex_t *indices_data;

  THCUNN_assertSameGPU(state, 3, input, output, indices);
  THArgCheck( input->nDimension == 2 || input->nDimension == 3, 2, "2D or 3D(batch mode) tensor expected");

  if (input->nDimension == 3)
  {
    dimT = 1;
    dimF = 2;
    batch = input->size[0];
  }
  THArgCheck( input->size[dimT] >= kW, 2, "input sequence smaller than kernel size");

  input = THCTensor_(newContiguous)(state, input);

  input_w = input->size[dimT];
  input_n = input->size[dimF];
  output_w = (input_w - kW) / dW + 1;

  if (input->nDimension == 2)
  {
    THCTensor_(resize2d)(state, output, output_w, input->size[dimF]);
    THCIndexTensor_(resize2d)(state, indices, output_w, input->size[dimF]);
  }
  else
  {
    THCTensor_(resize3d)(state, output, batch, output_w, input->size[dimF]);
    THCIndexTensor_(resize3d)(state, indices, batch, output_w, input->size[dimF]);
  }

  input_data = THCTensor_(data)(state, input);
  output_data = THCTensor_(data)(state, output);
  indices_data = THCIndexTensor_(data)(state, indices);

  dim3 blocks(batch);
  nthreads = (output_w / 32) * 32;
  if (output_w % 32 > 0) {
    nthreads += 32;
  }

  if (nthreads > TEMPORAL_MAX_POOLING_THREADS) {
    blocks.y = nthreads / TEMPORAL_MAX_POOLING_THREADS;
    if (nthreads % TEMPORAL_MAX_POOLING_THREADS > 0) {
      blocks.y += 1;
    }
    nthreads = TEMPORAL_MAX_POOLING_THREADS;
  }

  dim3 threads(nthreads);
  cunn_TemporalMaxPooling_updateOutputKernel <<< blocks, threads, 0, THCState_getCurrentStream(state) >>>(
      input_data, output_data, indices_data, input_w, input_n, output_w, kW, dW);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, input);

}

void THNN_(TemporalMaxPooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCIndexTensor *indices,
           int kW, int dW) {

  int dimT = 0; // Temporal dimension
  int dimF = 1; // Feature dimension

  int batch = 1;
  int input_w;
  int input_n;
  int output_w;
  int nthreads;

  real *gradInput_data;
  real *gradOutput_data;
  THCIndex_t *indices_data;

  THCUNN_assertSameGPU(state, 4, input, gradOutput, gradInput, indices);
  THArgCheck( input->nDimension == 2 || input->nDimension == 3, 2, "2D or 3D(batch mode) tensor expected");

  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(zero)(state, gradInput);

  if (input->nDimension == 3)
  {
    dimT = 1;
    dimF = 2;
    batch = input->size[0];
  }
  THArgCheck( input->size[dimT] >= kW, 2, "input sequence smaller than kernel size");

  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  input_w = input->size[dimT];
  input_n = input->size[dimF];
  output_w = (input_w - kW) / dW + 1;

  gradInput_data = THCTensor_(data)(state, gradInput);
  gradOutput_data = THCTensor_(data)(state, gradOutput);
  indices_data = THCIndexTensor_(data)(state, indices);

  dim3 blocks(batch);
  nthreads = (output_w / 32) * 32;
  if (output_w % 32 > 0) {
    nthreads += 32;
  }

  if (nthreads > TEMPORAL_MAX_POOLING_THREADS) {
    blocks.y = nthreads / TEMPORAL_MAX_POOLING_THREADS;
    if (nthreads % TEMPORAL_MAX_POOLING_THREADS > 0) {
      blocks.y += 1;
    }
    nthreads = TEMPORAL_MAX_POOLING_THREADS;
  }

  dim3 threads(nthreads);
  if (kW <= dW) {
    cunn_TemporalMaxPooling_updateGradInputKernel <<< blocks, threads, 0, THCState_getCurrentStream(state) >>>(
        gradInput_data, gradOutput_data, indices_data, input_w, input_n, output_w, kW, dW);
  } else {
    cunn_TemporalMaxPooling_updateGradInputKernelAtomic <<< blocks, threads, 0, THCState_getCurrentStream(state) >>>(
        gradInput_data, gradOutput_data, indices_data, input_w, input_n, output_w, kW, dW);
  }
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradOutput);

}

#endif
