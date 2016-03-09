#include "THCUNN.h"
#include "common.h"

#define TEMPORAL_MAX_POOLING_THREADS 1024

__global__ void cunn_TemporalMaxPooling_updateOutputKernel(float *input, float *output, float *indices, int input_w, int input_n, int output_w, int kW, int dW) {
  // Block idx is the batch index, thread idx + block idx y * MAX_THREADS is the time index
  float *input_data = input + blockIdx.x * input_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n * dW;
  float *output_data = output + blockIdx.x * output_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n;
  float *indices_data = indices + blockIdx.x * output_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n;

  int feat = 0;
  int time = 0;
  int max_time = input_n * kW;

  float max_value;
  float max_index = 0.0;

  if (threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS < output_w) {
    // For all features
    for (feat = 0; feat < input_n; ++feat) {
      max_value = -FLT_MAX;
      // For all values in the kernel space
      for (time = 0; time < max_time; time += input_n) {
        if (max_value < input_data[time + feat]) {
          max_value = input_data[time + feat];
          max_index = time / input_n;
        }
      }
      output_data[feat] = max_value;
      indices_data[feat] = (float)max_index;
    }
  }
}

__global__ void cunn_TemporalMaxPooling_updateGradInputKernel(float *gradInput, float *gradOutput, float *indices, int input_w, int input_n, int output_w, int kW, int dW) {
  // Block idx is the batch index, thread idx + block idx y * MAX_THREADS is the time index
  float *gradInput_data = gradInput + blockIdx.x * input_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n * dW;
  float *gradOutput_data = gradOutput + blockIdx.x * output_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n;
  float *indices_data = indices + blockIdx.x * output_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n;

  int feat = 0;

  if (threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS < output_w) {
    // For all features
    for (feat = 0; feat < input_n; ++feat) {
      gradInput_data[(int)indices_data[feat] * input_n + feat] += gradOutput_data[feat];
    }
  }
}

__global__ void cunn_TemporalMaxPooling_updateGradInputKernelAtomic(float *gradInput, float *gradOutput, float *indices, int input_w, int input_n, int output_w, int kW, int dW) {
  // Block idx is the batch index, thread idx + block idx y * MAX_THREADS is the time index
  float *gradInput_data = gradInput + blockIdx.x * input_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n * dW;
  float *gradOutput_data = gradOutput + blockIdx.x * output_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n;
  float *indices_data = indices + blockIdx.x * output_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n;

  int feat = 0;

  if (threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS < output_w) {
    // For all features
    for (feat = 0; feat < input_n; ++feat) {
      atomicAdd(&gradInput_data[(int)indices_data[feat] * input_n + feat], gradOutput_data[feat]);
    }
  }
}

void THNN_CudaTemporalMaxPooling_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *indices,
          int kW, int dW) {

  int dimT = 0; // Temporal dimension
  int dimF = 1; // Feature dimension

  int batch = 1;
  int input_w;
  int input_n;
  int output_w;
  int nthreads;

  float *input_data;
  float *output_data;
  float *indices_data;

  THCUNN_assertSameGPU(state, 3, input, output, indices);
  THArgCheck( input->nDimension == 2 || input->nDimension == 3, 2, "2D or 3D(batch mode) tensor expected");

  if (input->nDimension == 3)
  {
    dimT = 1;
    dimF = 2;
    batch = input->size[0];
  }
  THArgCheck( input->size[dimT] >= kW, 2, "input sequence smaller than kernel size");

  input = THCudaTensor_newContiguous(state, input);

  input_w = input->size[dimT];
  input_n = input->size[dimF];
  output_w = (input_w - kW) / dW + 1;

  if (input->nDimension == 2)
  {
    THCudaTensor_resize2d(state, output, output_w, input->size[dimF]);
    THCudaTensor_resize2d(state, indices, output_w, input->size[dimF]);
  }
  else
  {
    THCudaTensor_resize3d(state, output, batch, output_w, input->size[dimF]);
    THCudaTensor_resize3d(state, indices, batch, output_w, input->size[dimF]);
  }

  input_data = THCudaTensor_data(state, input);
  output_data = THCudaTensor_data(state, output);
  indices_data = THCudaTensor_data(state, indices);

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

  THCudaTensor_free(state, input);

}

void THNN_CudaTemporalMaxPooling_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *indices,
          int kW, int dW) {

  int dimT = 0; // Temporal dimension
  int dimF = 1; // Feature dimension

  int batch = 1;
  int input_w;
  int input_n;
  int output_w;
  int nthreads;

  float *gradInput_data;
  float *gradOutput_data;
  float *indices_data;

  THCUNN_assertSameGPU(state, 4, input, gradOutput, gradInput, indices);
  THArgCheck( input->nDimension == 2 || input->nDimension == 3, 2, "2D or 3D(batch mode) tensor expected");

  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_zero(state, gradInput);

  if (input->nDimension == 3)
  {
    dimT = 1;
    dimF = 2;
    batch = input->size[0];
  }
  THArgCheck( input->size[dimT] >= kW, 2, "input sequence smaller than kernel size");

  gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  input_w = input->size[dimT];
  input_n = input->size[dimF];
  output_w = (input_w - kW) / dW + 1;

  gradInput_data = THCudaTensor_data(state, gradInput);
  gradOutput_data = THCudaTensor_data(state, gradOutput);
  indices_data = THCudaTensor_data(state, indices);

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

  THCudaTensor_free(state, gradOutput);

}
